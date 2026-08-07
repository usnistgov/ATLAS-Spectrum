"""
iq_ingest.py: precompute a multi-resolution STFT spectrogram pyramid for
IQ captures (SigMF / TDMS / npy) into iq.duckdb. Resumable like psd_ingest.py.

Per capture: Hann-window STFT (nfft bins, hop = nfft), power in dBm (50 ohm),
fftshifted so row 0 = fc - fs/2. dB is quantized to uint8 with a FIXED per-
capture [qmin,qmax] so colours never drift across zoom. Level 0 = full STFT
resolution; each higher level max-pools time columns 2:1 down to <=1024 cols.
Blobs are frame-major chunks of <=512 columns (~0.5 MB at nfft=1024).

    py iq_ingest.py iqdata/mds2-3177 --dataset mds2-3177   # a folder (recursive)
    py iq_ingest.py path/to/capture.sigmf-meta             # a single capture
    py iq_ingest.py iqdata --limit 2                       # quick test
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb
import numpy as np

import cbrs_files
import sigmf_io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root (scripts live in ingest/)
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
IQ_DB = os.environ.get("IQ_DB") or os.path.join(DB_DIR, "iq.duckdb")

NFFT = 1024
CHUNK_COLS = 512          # columns per stored blob (nfft x 512 = 0.5 MB)
MAX_COARSE_COLS = 1024    # coarsest level fits one overview screen

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# The iq-tar entry is the samples file, not its .xml sidecar: the sidecar is
# found from the samples file, and matching both would ingest every capture twice.
CAPTURE_EXT = (".sigmf-meta", ".tdms", ".npy", ".complex1ch.float32")


def discover(root):
    """Capture files under root (or root itself), by extension."""
    return cbrs_files.walk_ext(root, CAPTURE_EXT)


def stft_db(cap, s0, nframes, win):
    """dB power for `nframes` hops starting at sample s0. (nframes, NFFT)."""
    x = cap.read(s0, s0 + nframes * NFFT)
    frames = x.reshape(nframes, NFFT) * win
    X = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    # mean-square volts per bin -> dBm across 50 ohm (readout shows real dBm)
    pwr = (np.abs(X) ** 2) / (np.sum(win.astype(np.float64) ** 2) * NFFT)
    return 10.0 * np.log10(np.maximum(pwr / 50.0 * 1000.0, 1e-30))


def scan_scale(cap, win):
    """Sparse pre-pass -> fixed quantization range for the whole capture."""
    total = cap.n_samples // NFFT
    if total < 1:
        raise ValueError(f"only {cap.n_samples} sample(s); an STFT needs at "
                         f"least {NFFT}. This file is too short to render, or "
                         "its .sigmf-data is truncated.")
    step = max(1, total // 64)
    samp = []
    for f0 in range(0, total, step):
        n = min(32, total - f0)
        samp.append(stft_db(cap, f0 * NFFT, n, win))
    db = np.concatenate(samp)
    return float(np.percentile(db, 0.1) - 6.0), float(np.percentile(db, 99.9) + 6.0)


def ingest_capture(connection, path, dataset):
    cap = sigmf_io.open_capture(path)
    cid = cap.name
    if cap.n_samples < NFFT:
        raise ValueError(f"only {cap.n_samples} sample(s); an STFT needs at "
                         f"least {NFFT}. This file is too short to render, or "
                         "its .sigmf-data is truncated.")
    if connection.execute("SELECT 1 FROM iq_meta WHERE id=?", [cid]).fetchone():
        print(f"  skip (already ingested): {cid}")
        return False
    if cap.center_freq == 0.0:
        print(f"  NOTE {cid}: no center frequency in metadata; Y axis will be baseband")
    win = np.hanning(NFFT).astype(np.float32)
    qmin, qmax = scan_scale(cap, win)
    ncols0 = cap.n_samples // NFFT
    print(f"  {cid}: fs={cap.sample_rate/1e6:.3g} Msps fc={cap.center_freq/1e6:.6g} MHz "
          f"dur={cap.duration:.3f}s cols={ncols0} q=[{qmin:.1f},{qmax:.1f}] dBm")
    connection.execute("DELETE FROM iq_stft WHERE id=?", [cid])

    # ---- level 0: stream the full STFT in CHUNK_COLS blocks ----
    t0 = time.time()
    prev_ncols = ncols0
    for c0 in range(0, ncols0, CHUNK_COLS):
        n = min(CHUNK_COLS, ncols0 - c0)
        db = stft_db(cap, c0 * NFFT, n, win)
        q = np.clip(np.round((db - qmin) / (qmax - qmin) * 255.0), 0, 255).astype(np.uint8)
        connection.execute("INSERT INTO iq_stft VALUES (?,?,?,?,?)",
                    [cid, 0, c0, n, q.tobytes()])
    print(f"    level 0: {ncols0} cols in {time.time()-t0:.1f}s")

    # ---- higher levels: 2:1 max-pool of the previous level's columns ----
    level = 0
    while prev_ncols > MAX_COARSE_COLS:
        level += 1
        ncols = prev_ncols // 2
        for c0 in range(0, ncols, CHUNK_COLS):
            n = min(CHUNK_COLS, ncols - c0)
            rows = connection.execute(
                "SELECT col0, ncols, chunk FROM iq_stft WHERE id=? AND level=? "
                "AND col0 < ? AND col0 + ncols > ? ORDER BY col0",
                [cid, level - 1, (c0 + n) * 2, c0 * 2]).fetchall()
            src = np.concatenate([
                np.frombuffer(r[2], dtype=np.uint8).reshape(r[1], NFFT) for r in rows])
            off = c0 * 2 - rows[0][0]
            pair = src[off:off + n * 2].reshape(n, 2, NFFT).max(axis=1)
            connection.execute("INSERT INTO iq_stft VALUES (?,?,?,?,?)",
                        [cid, level, c0, n, pair.tobytes()])
        prev_ncols = ncols

    # ---- display scale from the coarsest level (stable colours) ----
    rows = connection.execute("SELECT chunk FROM iq_stft WHERE id=? AND level=?",
                       [cid, level]).fetchall()
    coarse = np.concatenate([np.frombuffer(r[0], dtype=np.uint8) for r in rows])
    dbv = qmin + coarse.astype(np.float32) / 255.0 * (qmax - qmin)
    vmin, vmax = float(np.percentile(dbv, 2)), float(np.percentile(dbv, 98))
    if vmax - vmin < 6.0:
        vmax = vmin + 6.0

    connection.execute("INSERT INTO iq_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [cid, dataset, cap.name, os.path.abspath(cap.path),
                 cap.center_freq, cap.sample_rate, cap.duration, cap.n_samples,
                 NFFT, NFFT, NFFT, level + 1, qmin, qmax,
                 round(vmin, 1), round(vmax, 1)])
    print(f"    {level+1} levels, done in {time.time()-t0:.1f}s")
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: py iq_ingest.py <capture-file-or-folder> "
                 "[--dataset name] [--limit N]")
    root = sys.argv[1]
    dataset = (sys.argv[sys.argv.index("--dataset") + 1]
               if "--dataset" in sys.argv else os.path.basename(os.path.normpath(root)))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    paths = discover(root)
    if limit:
        paths = paths[:limit]
    print(f"{len(paths)} capture file(s) under {root} -> {IQ_DB}")

    connection = duckdb.connect(IQ_DB)
    connection.execute("""CREATE TABLE IF NOT EXISTS iq_meta (
        id VARCHAR PRIMARY KEY, dataset VARCHAR, name VARCHAR, path VARCHAR,
        fc DOUBLE, fs DOUBLE, duration DOUBLE, n_samples BIGINT,
        nfft INT, nfreq INT, hop INT, nlevels INT,
        qmin DOUBLE, qmax DOUBLE, vmin DOUBLE, vmax DOUBLE)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS iq_stft (
        id VARCHAR, level INT, col0 BIGINT, ncols INT, chunk BLOB)""")

    done = err = 0
    for p in paths:
        try:
            if ingest_capture(connection, p, dataset):
                done += 1
        except Exception as e:
            err += 1
            print(f"  ERR {os.path.basename(p)}: {e}")
    connection.close()
    size = os.path.getsize(IQ_DB) / 1e6
    print(f"\nDone: {done} ingested, {err} errors. {IQ_DB} = {size:.1f} MB")
    if err and done == 0:
        # Exiting 0 here is how a failed ingest used to look like a success to
        # anything driving this script.
        print(f"Nothing was ingested: all {err} capture(s) failed to read.",
              file=sys.stderr)
        return 1
    if err:
        print(f"{err} capture(s) failed; re-run to retry them.", file=sys.stderr)
    if done:
        print("next: python serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
