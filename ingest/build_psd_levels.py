"""build_psd_levels.py: a coarse-time pyramid for the PSD layer.

The PSD layer stores one spectrum per capture. Drawing a WIDE time window from
that means reading and zlib-inflating every chunk in range -- on a 200-day
window that is hundreds of megabytes of inflation to fill a few thousand pixel
columns, and it measured 18 s. serve.py already thinned the captures it kept
(one in eight at that width), so nearly all of that inflation was thrown away.

This builds what the summary layer has always had: max spectra over fixed time
buckets, so a wide window reads a few thousand small rows instead.

    psd_lvl(sensor, bucket, t, smax)     smax = 2250 uint8 bytes, the bin-wise
                                         max over [t, t + bucket)

Two things worth being precise about:

  * It is not a loss of fidelity at those widths -- it is a gain. A column that
    used to be the max over one capture in eight is now the max over EVERY
    capture in the bucket. Closer to the max the data actually holds, not
    further from it.
  * `max` is what makes this sound. Max is associative, so the 6 h level is the
    bin-wise max of six 1 h rows and the 1 d level the max of four 6 h rows --
    each derived exactly from the one below it, no re-reading. A mean or median
    layer could NOT be built this way.

WHAT `--stat median` AND `--stat mean` ACTUALLY BUILD, because the name misleads:
a level row is the MAX OVER THE BUCKET of the per-sweep medians (or means), not
the median or mean over the bucket. Pooling is max for every statistic, here and
in serve.py's own per-pixel pooling -- the stored per-capture values are the
medians, and keeping the strongest capture visible is the display convention the
capture path uses too. So it is consistent, not a bug, but it is not the
statistic the button is labelled with, and the gap is large: measured on real
exports at the 1-day bucket, the median layer reads +2.9 to +10.6 dB above the
true day median typically, worst bin +21.9 dB, and it grows with bucket width
(+1.6 dB at 10 min, +4.4 at 1 h, +7.8 at 6 h, +10.6 at 1 day for NIT median).

Turning the pyramid off does not restore median semantics -- the capture path it
falls back to is a max over a thinned subsample, measured at +8.5 dB on the same
window, so the pyramid accounts for only ~2 dB of that. And where the server
reads every capture (narrow windows) it refuses the level entirely, so the exact
answer is never degraded. The number on screen is exact at PSD zoom; it is a
max-of-medians when zoomed out.

Additive and INCREMENTAL: each sensor records how far through time it has been
summarised, so running this again after next month's ingest reads only the new
captures instead of the whole sensor. Re-run it after every ingest -- serve.py
will not draw from a level beyond the range it actually covers, so forgetting to
means wide windows quietly go back to being slow, not to being wrong.

Every existing table is left untouched. Stop serve.py first -- DuckDB allows one
writer at a time.

    py build_psd_levels.py                  # build or top up every sensor
    py build_psd_levels.py --sensor HU
    py build_psd_levels.py --rebuild        # from scratch
"""
import argparse
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb
import numpy as np

import chunk_io                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
PSD_DB = os.environ.get("PSD_DB") or os.path.join(DB_DIR, "psd.duckdb")

NF = 2250
# Finest bucket first; each coarser one an exact multiple of the one before, so
# it can be derived from it by max instead of re-read. serve.py only reaches for
# a level once a pixel column is at least one bucket wide, so the finest level
# sets how narrow a window still gets the speed-up: 10 min covers spans past
# ~7 days on a 1300-column plot, 1 h only past ~40 days. Below that the captures
# are already sub-500 ms and are read unchanged.
#
# The 10 min level is the bulk of the size (~65 MB per sensor-200-days, against
# ~11 MB for all three coarser ones together). --coarse skips it if the disk
# matters more than mid-range zoom.
BASE = 600
DERIVED = [3600, 21600, 86400]
LEVELS = [BASE] + DERIVED

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_tables = chunk_io.tables      # one definition, in the module that owns the schema


def _iter_captures(connection, sensor, kind, since):
    """(times, spectra) batches for one sensor, whichever schema is on disk.

    `since` skips whole chunks that end before it; the caller still masks the
    individual captures, because a chunk straddling the boundary carries both.
    """
    if kind == "chunk":
        cur = connection.execute("SELECT n, times, specs FROM psd_chunk WHERE sensor=? "
                          "AND t1>=? ORDER BY t0", [sensor, since])
        while True:
            rows = cur.fetchmany(64)
            if not rows:
                return
            for n, tb, sb in rows:
                ts = np.frombuffer(zlib.decompress(tb), np.float64)
                mat = np.frombuffer(zlib.decompress(sb), np.uint8).reshape(n, NF)
                yield ts, mat
    else:
        cur = connection.execute("SELECT t, spec FROM psd WHERE sensor=? AND t>=? "
                          "ORDER BY t", [sensor, since])
        while True:
            rows = cur.fetchmany(4096)
            if not rows:
                return
            ts = np.array([r[0] for r in rows], np.float64)
            mat = np.stack([np.frombuffer(r[1], np.uint8) for r in rows])
            yield ts, mat


def build_sensor(connection, sensor, kind, since):
    """-> (rows written, latest capture time seen), from `since` onwards."""
    t_start = time.time()
    # Bin-wise max per BASE bucket, accumulated in memory. One bucket is 2250
    # bytes; a year at 1 h is 8760 of them, under 20 MB, so a dict is fine and
    # avoids a second pass over the captures.
    buckets = {}
    ncap = 0
    tmax = since
    for ts, mat in _iter_captures(connection, sensor, kind, since):
        keep = ts >= since
        if not keep.any():
            continue
        ts, mat = ts[keep], mat[keep]
        ncap += len(ts)
        tmax = max(tmax, float(ts[-1]))
        keys = (ts // BASE).astype(np.int64)
        for k in np.unique(keys):
            sel = mat[keys == k]
            m = sel.max(axis=0)
            prev = buckets.get(k)
            buckets[k] = m if prev is None else np.maximum(prev, m)
    if not buckets:
        return 0, since

    written = 0
    # Only the range being rebuilt is cleared. `since` is aligned down to the
    # COARSEST bucket by the caller, so every coarse bucket that gets rewritten
    # here is fully covered by the base buckets recomputed above -- otherwise a
    # coarse row spanning the boundary would be replaced by one summarising only
    # the new half of its own time range.
    connection.execute("DELETE FROM psd_lvl WHERE sensor=? AND t>=?", [sensor, int(since)])
    cur_level = {k * BASE: v for k, v in buckets.items()}
    connection.executemany("INSERT INTO psd_lvl VALUES (?,?,?,?)",
                    [(sensor, BASE, int(t), v.tobytes())
                     for t, v in sorted(cur_level.items())])
    written += len(cur_level)

    # Coarser levels by exact max of the level below -- no re-reading.
    prev_bucket = BASE
    for bucket in DERIVED:
        assert bucket % prev_bucket == 0, "levels must nest exactly"
        nxt = {}
        for t, v in cur_level.items():
            k = (t // bucket) * bucket
            cur = nxt.get(k)
            nxt[k] = v if cur is None else np.maximum(cur, v)
        connection.executemany("INSERT INTO psd_lvl VALUES (?,?,?,?)",
                        [(sensor, bucket, int(t), v.tobytes())
                         for t, v in sorted(nxt.items())])
        written += len(nxt)
        cur_level, prev_bucket = nxt, bucket

    log(f"  {sensor}: {ncap:,} captures -> {written:,} level rows "
        f"in {time.time()-t_start:.0f}s")
    return written, tmax


def main():
    global BASE, DERIVED, LEVELS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sensor", default=None, help="one sensor (default: all)")
    ap.add_argument("--rebuild", action="store_true",
                    help="redo sensors already built")
    ap.add_argument("--coarse", action="store_true",
                    help="skip the finest (10 min) level: much smaller on disk, "
                         "but mid-range spans then read the captures as before")
    ap.add_argument("--stat", default="max", choices=("max", "median", "mean"),
                    help="which statistic's PSD database to summarise "
                         "(median/mean live in psd_<stat>.duckdb)")
    args = ap.parse_args()

    global PSD_DB
    if args.stat != "max":
        PSD_DB = os.path.join(os.path.dirname(PSD_DB) or ".",
                              f"psd_{args.stat}.duckdb")
        log(f"stat '{args.stat}' -> {os.path.basename(PSD_DB)}")

    if args.coarse:
        BASE, DERIVED = DERIVED[0], DERIVED[1:]
        LEVELS = [BASE] + DERIVED

    if not os.path.exists(PSD_DB):
        sys.exit(f"{PSD_DB} not found. Build it with ingest/psd_ingest.py first.")
    try:
        connection = duckdb.connect(PSD_DB)
    except Exception as e:
        sys.exit(f"cannot open {os.path.basename(PSD_DB)} for writing: {e}\n"
                 "  Stop serve.py (and any other reader) and run this again -- "
                 "DuckDB allows one writer at a time.")

    have = _tables(connection)
    kind = ("chunk" if "psd_chunk" in have
            else "rows" if "psd" in have else None)
    if kind is None:
        sys.exit(f"{os.path.basename(PSD_DB)} has no PSD table to summarise.")

    connection.execute("""CREATE TABLE IF NOT EXISTS psd_lvl (
        sensor VARCHAR, bucket INTEGER, t BIGINT, smax BLOB)""")
    # built_through is what makes a re-run incremental. An older build wrote this
    # table with the sensor name alone, which made a re-run after new data skip
    # the sensor and leave the levels short of the captures; drop that shape and
    # rebuild rather than guess how far it had got.
    cols = ({r[1] for r in connection.execute(
                "PRAGMA table_info(psd_lvl_done)").fetchall()}
            if "psd_lvl_done" in _tables(connection) else set())
    if cols and "built_through" not in cols:
        log("  older progress table found; rebuilding the levels once")
        connection.execute("DROP TABLE psd_lvl_done")
        connection.execute("DELETE FROM psd_lvl")
    connection.execute("CREATE TABLE IF NOT EXISTS psd_lvl_done ("
                "sensor VARCHAR, built_through DOUBLE)")

    src = "psd_chunk" if kind == "chunk" else "psd"
    sensors = [r[0] for r in connection.execute(
        f"SELECT DISTINCT sensor FROM {src} ORDER BY 1").fetchall()]
    if args.sensor:
        if args.sensor not in sensors:
            sys.exit(f"no PSD for sensor '{args.sensor}'. Have: "
                     + ", ".join(sensors))
        sensors = [args.sensor]
    if args.rebuild:
        connection.execute("DELETE FROM psd_lvl_done"
                    + ("" if args.sensor is None else " WHERE sensor=?"),
                    [] if args.sensor is None else [args.sensor])

    done = {r[0]: r[1] for r in connection.execute(
        "SELECT sensor, built_through FROM psd_lvl_done").fetchall()}
    log(f"{os.path.basename(PSD_DB)} ({kind} schema): {len(sensors)} sensor(s), "
        f"levels {', '.join(str(b) for b in LEVELS)}s")
    total = 0
    coarsest = max(LEVELS)
    for s in sensors:
        prev = done.get(s)
        # Realign to the coarsest bucket boundary and redo from there, so the one
        # partial bucket at the end of the last run is completed rather than left
        # summarising half its own time range.
        since = -1e18 if prev is None else (prev // coarsest) * coarsest
        n, tmax = build_sensor(connection, s, kind, since)
        if n == 0:
            log(f"  {s}: nothing new" if prev is not None else f"  {s}: no captures")
            continue
        connection.execute("DELETE FROM psd_lvl_done WHERE sensor=?", [s])
        connection.execute("INSERT INTO psd_lvl_done VALUES (?,?)", [s, float(tmax)])
        total += n
    rows = connection.execute("SELECT count(*) FROM psd_lvl").fetchone()[0]
    connection.close()
    log(f"DONE: {total:,} row(s) added this run; psd_lvl holds {rows:,}. "
        f"DB is now {os.path.getsize(PSD_DB)/1e9:.2f} GB")
    log("serve.py picks the levels up automatically on its next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
