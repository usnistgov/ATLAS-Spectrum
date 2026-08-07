"""verify_csv_1to1.py: prove a PSD database holds exactly what the CSV said.

The question this answers is narrow and worth stating precisely: for a given
sensor and statistic, is every number the viewer can draw the same number the
NASCTN export carried, at the same time, at the same frequency?

It is checked several ways, because each catches a different kind of wrong:

  values   dequantize the byte the database stores and compare it to the CSV's
           own float, for EVERY cell. This is the only check that is
           formula-agnostic -- it does not care how the ingest quantizes, only
           that the stored byte means the CSV's number. The bound is half a
           quantization step, which is the most a correct round-to-nearest can
           be off by.
  bytes    recompute the expected byte from the CSV with exact rational
           arithmetic (fractions.Fraction, no floating point) and require the
           stored byte to equal it. Catches a biased quantizer that "values"
           would forgive, because at a value landing exactly halfway between two
           bytes BOTH bytes are within half a step and the value check cannot
           tell them apart. Those halfway values are enumerated and checked
           against the tie rule specifically, so a round-half-up or
           round-half-down quantizer fails here.
  routing  the database must match its OWN statistic's CSV and must NOT match
           the other statistics' CSVs. Without this, an ingest that put max data
           into the mean database would pass everything else.
  geometry timestamps to the microsecond, capture order, bin order, and the
           frequency each bin claims -- against the frequencies in the CSV's own
           header row. A one-bin shift or a transposed matrix passes neither.
  levels   every coarse-pyramid row is recomputed from the CSV floats, not from
           the database's own fine table, so a pyramid that is internally
           consistent with wrong data still fails.

Two modes:

    # build an isolated database from the exports under --root and check it
    py examples/verify_csv_1to1.py --csv <file>.csv --sensor HU --stat max

    # check a database you already have, without writing to it
    py examples/verify_csv_1to1.py --csv <file>.csv --sensor HU --stat max \
        --db C:\\Users\\you\\ATLAS\\psd.duckdb

`--db` is the one that matters for an existing install: it opens the file
read-only, so it is safe to run against the databases the viewer is serving.

Exit code is 0 only if every check passed. A check that had nothing to look at
fails rather than passes -- see test_verify_1to1.py, which damages a real
database every way the ingest could plausibly get one wrong and requires each to
be caught by a check that owns it, not incidentally by an unrelated one.
"""

import argparse
import csv
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ingest"))

import duckdb                                            # noqa: E402

NF = 2250
F0 = 3530040000.0
DF = 80000.0
QMIN, QMAX = -180.0, -90.0
STEP = (QMAX - QMIN) / 255.0            # 0.352941 dB per count
# The most a correct nearest-rounding can miss by. The 1e-9 is not slack: on real
# data thousands of cells exceed STEP/2 by ~1e-14 purely from float64 evaluation
# of the dequantization, so without it a CORRECT database fails. Verified: it is
# far smaller than one count (0.353 dB), so it cannot hide a real error.
HALF = STEP / 2.0 + 1e-9
# A CSV value lands exactly halfway between two bytes when 255*(v+180)/90 is a
# half-integer, i.e. when v+180 is an odd multiple of 3: v = -177, -171, ... -93.
# At those values both neighbouring bytes are within half a step, so the value
# check is blind and only the tie rule decides. Enumerated here so they can be
# checked exhaustively instead of hoped over.
TIE_VALUES = np.array([-180.0 + 3.0 * k for k in range(1, 30, 2)])   # 15 values


def is_tie(v):
    return np.isclose(v[..., None], TIE_VALUES, atol=1e-9, rtol=0).any(axis=-1)


# ---------------------------------------------------------------- reading the CSV

def read_csv_independently(path):
    """(stamps[n] as int microseconds, freqs[2250], values[n,2250] float64).

    Deliberately not DuckDB and not numpy's parser: the ingest reads these files
    with `read_csv_auto`, so reading them the same way here would make a parser
    bug invisible. Python's csv module and float() are a separate path.
    """
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", newline="", errors="replace") as fh:
        r = csv.reader(fh)
        head = next(r)
        freqs = np.array([float(x) for x in head[1:]], dtype=np.float64)
        stamps, rows = [], []
        for line in r:
            if not line or not line[0].strip():
                continue
            stamps.append(parse_stamp_us(line[0]))
            rows.append([float(x) if x.strip() not in ("", "nan", "NaN", "NULL")
                         else np.nan for x in line[1:]])
    return (np.array(stamps, dtype=np.int64), freqs,
            np.array(rows, dtype=np.float64))


def parse_stamp_us(s):
    """'2025-07-09 00:02:39.258000+00:00' -> epoch microseconds (int).

    Hand-rolled rather than np.datetime64 for the same reason as above, and it
    keeps integer microseconds throughout so no float rounding enters the
    timestamp comparison. A UTC offset is HONOURED, not discarded -- an export
    stamped +05:30 names a different instant than the same digits in UTC.
    """
    s = s.strip().replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    body, off = s, 0
    tail = s[10:]
    for sign in ("+", "-"):
        if sign in tail:
            body = s[:10] + tail.split(sign)[0]
            hh, _, mm = tail.split(sign)[1].partition(":")
            off = (int(hh) * 3600 + int(mm or 0) * 60) * (1 if sign == "+" else -1)
            break
    date, _, clock = body.strip().partition(" ")
    y, mo, d = (int(x) for x in date.split("-"))
    hh, mm, ss = (clock.split(":") + ["0", "0", "0"])[:3]
    sec, _, frac = ss.partition(".")
    days = days_from_civil(y, mo, d)
    total = days * 86400 + int(hh) * 3600 + int(mm) * 60 + int(sec) - off
    return total * 1_000_000 + int((frac + "000000")[:6])


def days_from_civil(y, m, d):
    """Days since 1970-01-01, Howard Hinnant's algorithm. No datetime import."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


# ------------------------------------------------------- reading the database

def db_kind(c):
    have = {r[0] for r in c.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if "psd_chunk" in have and c.execute(
            "SELECT count(*) FROM psd_chunk").fetchone()[0]:
        return "chunk"
    if "psd" in have and c.execute("SELECT count(*) FROM psd").fetchone()[0]:
        return "rows"
    return "empty"


def db_read(c, sensor, kind):
    """(times[n] float64 epoch seconds, spec[n,2250] uint8), capture order.

    Own chunk reader rather than serve.py's, so a bug in serve.py's reader
    cannot hide by being used on both sides of the comparison.
    """
    if kind == "rows":
        rows = c.execute("SELECT t, spec FROM psd WHERE sensor=? ORDER BY t",
                         [sensor]).fetchall()
        if not rows:
            return np.empty(0), np.empty((0, NF), np.uint8)
        ts = np.array([r[0] for r in rows], dtype=np.float64)
        sp = np.stack([np.frombuffer(r[1], dtype=np.uint8) for r in rows])
        return ts, sp
    tparts, sparts = [], []
    for t0, times, specs, n in c.execute(
            "SELECT t0, times, specs, n FROM psd_chunk WHERE sensor=? ORDER BY t0",
            [sensor]).fetchall():
        ts = np.frombuffer(zlib.decompress(times), dtype=np.float64)
        sp = np.frombuffer(zlib.decompress(specs), dtype=np.uint8)
        if sp.size != len(ts) * NF:
            raise AssertionError(f"chunk at t0={t0} holds {sp.size} bytes for "
                                 f"{len(ts)} captures ({n} claimed)")
        tparts.append(ts)
        sparts.append(sp.reshape(len(ts), NF))
    if not tparts:
        return np.empty(0), np.empty((0, NF), np.uint8)
    ts = np.concatenate(tparts)
    sp = np.concatenate(sparts)
    order = np.argsort(ts, kind="stable")
    return ts[order], sp[order]


# ------------------------------------------------------------------- the checks

class Report:
    def __init__(self, label):
        self.label, self.rows, self.bad = label, [], 0

    def check(self, ok, name, detail=""):
        self.rows.append((bool(ok), name, detail))
        if not ok:
            self.bad += 1
        return bool(ok)

    def text(self, verbose=True):
        out = [f"{'PASS' if not self.bad else 'FAIL'}  {self.label}"]
        for ok, name, detail in self.rows:
            if verbose or not ok:
                out.append(f"    {'ok  ' if ok else 'FAIL'} {name}"
                           + (f"  -- {detail}" if detail else ""))
        return "\n".join(out)


def expected_byte_exact(p):
    """The byte a correct quantizer must store for CSV value p, exactly.

    Fraction arithmetic, so there is no floating-point step anywhere in the
    reference: q = round(255 * (p - QMIN) / (QMAX - QMIN)), clamped to 0..255,
    with halves rounded to even -- what both numpy's round and Python's round()
    do, and the only tie rule that is unbiased.
    """
    if p != p:                                    # NaN
        return None
    x = Fraction(repr(float(p)))
    v = (x - Fraction(-180)) * 255 / Fraction(90)
    if v <= 0:
        return 0
    if v >= 255:
        return 255
    lo = v.numerator // v.denominator
    rem = v - lo
    if rem > Fraction(1, 2):
        return lo + 1
    if rem < Fraction(1, 2):
        return lo
    return lo if lo % 2 == 0 else lo + 1          # tie -> even


def quantize_reference(vals):
    """Vectorized reference quantization, for the whole matrix at once."""
    with np.errstate(invalid="ignore"):
        return np.clip(np.round((vals - QMIN) / (QMAX - QMIN) * 255.0),
                       0, 255).astype(np.uint8)


def verify(csv_path, sensor, stat, dbpath, rng, samples, full_bytes, label,
           cross=(), strict_counts=False):
    """Verify one export against one database. `cross` = the other statistics'
    CSVs for the same sensor and day, which the database must NOT match."""
    rep = Report(label)
    stamps, freqs, vals = read_csv_independently(csv_path)
    n = len(stamps)
    rep.check(n > 0, "csv has captures", f"{n} rows")
    rep.check(vals.shape[1] == NF, "csv bin count",
              f"{vals.shape[1]} value columns (want {NF})")
    if n == 0 or vals.shape[1] != NF:
        return rep, None

    # -- NaN policy. A blank or NULL cell has no correct byte, and the ingest
    #    used to turn one into 255 -- the loudest value on the colour scale, an
    #    invented emitter where the data was silent. So a non-finite cell is a
    #    failure here rather than a masked-out exemption.
    nonfinite = int(np.count_nonzero(~np.isfinite(vals)))
    rep.check(nonfinite == 0, "csv holds no blank / NaN value cells",
              f"{nonfinite} non-finite cell(s)")

    # -- frequency axis, straight out of the CSV's own header
    want_f = F0 + np.arange(NF) * DF
    if len(freqs) == NF:
        dfmax = float(np.max(np.abs(freqs - want_f)))
        rep.check(dfmax < 1.0, "csv header frequencies match F0 + i*DF",
                  f"max |delta| = {dfmax:.3g} Hz over {NF} bins "
                  f"({freqs[0]:.0f} .. {freqs[-1]:.0f} Hz)")
    else:
        rep.check(False, "csv header frequencies", f"{len(freqs)} labels")

    c = duckdb.connect(dbpath, read_only=True)
    try:
        kind = db_kind(c)
        rep.check(kind in ("rows", "chunk"), "database has psd data", kind)
        dts, dsp = db_read(c, sensor, kind)

        # -- the CSV's captures must all be present, compared in integer
        #    microseconds (float64 resolves ~0.4 us at these epochs, so allow 1).
        want_us = stamps
        have_us = np.rint(dts * 1e6).astype(np.int64)
        idx = {}
        for i, v in enumerate(have_us):
            idx.setdefault(int(v), i)
        pos, missing = [], []
        for v in want_us:
            for d in (0, 1, -1):
                if int(v) + d in idx:
                    pos.append(idx[int(v) + d])
                    break
            else:
                missing.append(int(v))
        rep.check(not missing, "every csv capture is stored (to the microsecond)",
                  f"{n - len(missing)}/{n} present"
                  + (f"; first missing {missing[0]}" if missing else ""))
        if missing:
            return rep, None
        pos = np.array(pos, dtype=np.int64)
        rep.check(bool(np.all(np.diff(pos) > 0)),
                  "stored order matches csv order",
                  "strictly increasing" if np.all(np.diff(pos) > 0) else "reordered")
        rep.check(len(set(int(v) for v in have_us)) == len(have_us),
                  "no duplicate capture instants",
                  f"{len(have_us)} stored, {len(set(int(v) for v in have_us))} distinct")
        if strict_counts:
            # Only meaningful when the database was built from exactly the files
            # being verified: then a capture the CSV does not contain is spurious,
            # and containment alone would let a duplicated chunk through.
            rep.check(len(dts) == n, "database holds exactly the csv's captures",
                      f"{len(dts)} stored vs {n} in the csv")

        got = dsp[pos]                       # (n, NF) aligned to the CSV rows

        # -- CHECK 1: values. Formula-agnostic 1:1 test, every cell.
        deq = QMIN + got.astype(np.float64) * (QMAX - QMIN) / 255.0
        finite = np.isfinite(vals)
        inrange = finite & (vals >= QMIN) & (vals <= QMAX)
        err = np.abs(deq - vals)
        worst = float(np.max(err[inrange])) if inrange.any() else float("nan")
        nbad = int(np.count_nonzero(inrange & (err > HALF)))
        rep.check(nbad == 0 and inrange.any(),
                  "every in-range value round-trips within half a step",
                  f"{int(inrange.sum()):,} cells, worst |delta| = {worst:.6f} dB "
                  f"(limit {HALF:.6f}), {nbad} over")
        # Clipping is reported unconditionally, with its count, so "0 cells" is
        # visible rather than the row silently not appearing.
        lo = finite & (vals < QMIN)
        hi = finite & (vals > QMAX)
        rep.check(not lo.any() or bool(np.all(got[lo] == 0)),
                  "below-range values clip to 0", f"{int(lo.sum())} cell(s)")
        rep.check(not hi.any() or bool(np.all(got[hi] == 255)),
                  "above-range values clip to 255",
                  f"{int(hi.sum())} cell(s)"
                  + (f", worst {float(np.max(np.abs(deq[hi] - vals[hi]))):.3f} dB "
                     f"beyond the quantization ceiling" if hi.any() else ""))
        bias = float(np.mean((deq - vals)[inrange])) if inrange.any() else 0.0
        rep.check(abs(bias) < STEP / 4, "quantizer is unbiased",
                  f"mean signed error {bias:+.6f} dB")

        # -- CHECK 2a: every byte against the vectorized reference.
        ref = quantize_reference(vals)
        diff = int(np.count_nonzero((ref != got) & finite))
        rep.check(diff == 0, "every stored byte equals the reference quantization",
                  f"{int(finite.sum()):,} cells, {diff} differing")

        # -- CHECK 2b: the tie values, exhaustively. At these the value check is
        #    structurally blind, so this is the only thing standing between a
        #    round-half-up (or half-down) quantizer and a clean pass.
        tie = finite & is_tie(vals)
        ntie = int(np.count_nonzero(tie))
        if ntie:
            ii, jj = np.nonzero(tie)
            wrong = [(int(a), int(b), float(vals[a, b]), int(got[a, b]),
                      expected_byte_exact(vals[a, b]))
                     for a, b in zip(ii, jj)
                     if int(got[a, b]) != expected_byte_exact(vals[a, b])]
            rep.check(not wrong, "halfway values use round-half-to-even",
                      f"{ntie:,} halfway cell(s) checked exhaustively, "
                      f"{len(wrong)} wrong"
                      + (f"; e.g. row {wrong[0][0]} bin {wrong[0][1]} "
                         f"csv={wrong[0][2]} stored={wrong[0][3]} "
                         f"want={wrong[0][4]}" if wrong else ""))
        else:
            rep.check(True, "halfway values use round-half-to-even",
                      "no halfway values in this export")

        # -- CHECK 2c: exact rational arithmetic on a sample, as a guard against
        #    the vectorized reference sharing a bug with the ingest.
        k = min(full_bytes, n * NF)
        rep.check(k > 0, "exact-rational sample is non-empty", f"{k} cells")
        flat_i = rng.choice(n * NF, size=k, replace=False)
        mism, compared = [], 0
        for f in flat_i:
            i, j = divmod(int(f), NF)
            want = expected_byte_exact(vals[i, j])
            if want is None:
                continue
            compared += 1
            if int(got[i, j]) != want:
                mism.append((i, j, float(vals[i, j]), int(got[i, j]), want))
        rep.check(not mism and compared > 0,
                  "stored byte equals exact-rational reference",
                  f"{compared:,} cells compared with Fraction arithmetic, "
                  f"{len(mism)} mismatched"
                  + (f"; e.g. row {mism[0][0]} bin {mism[0][1]} csv={mism[0][2]} "
                     f"stored={mism[0][3]} want={mism[0][4]}" if mism else ""))

        # -- CHECK 3: named sample points, printed so they can be eyeballed.
        picks = []
        cand = [(0, 0), (0, NF - 1), (n - 1, 0), (n - 1, NF - 1), (n // 2, NF // 2)]
        am = int(np.argmax(np.where(finite, vals, -np.inf)))
        cand.append((am // NF, am % NF))
        for _ in range(samples):
            cand.append((int(rng.integers(n)), int(rng.integers(NF))))
        for i, j in cand:
            picks.append((i, j, float(vals[i, j]), int(got[i, j]),
                          float(deq[i, j]), F0 + j * DF, int(stamps[i])))
        okp = all(abs(p[4] - p[2]) <= HALF for p in picks
                  if QMIN <= p[2] <= QMAX and p[2] == p[2])
        rep.check(okp and bool(picks), "named sample points agree",
                  f"{len(picks)} points, worst "
                  f"{max(abs(p[4] - p[2]) for p in picks if p[2] == p[2]):.6f} dB")

        # -- CHECK 4: routing. This database must NOT be one of the others.
        for other in cross:
            try:
                _s2, _f2, v2 = read_csv_independently(other)
            except Exception as e:                              # noqa: BLE001
                rep.check(False, f"cross-check readable: {os.path.basename(other)}",
                          str(e))
                continue
            m = min(len(v2), n)
            fin2 = np.isfinite(v2[:m]) & finite[:m]
            e2 = np.abs(deq[:m] - v2[:m])
            worst2 = float(np.max(e2[fin2])) if fin2.any() else 0.0
            rep.check(worst2 > 1.0,
                      f"does NOT match {os.path.basename(other)}",
                      f"worst |delta| {worst2:.3f} dB "
                      f"(must exceed 1 dB or the layers are interchangeable)")

        # -- CHECK 5: metadata the server reads.
        m = c.execute("SELECT f0, df, nf, qmin, qmax, t_min, t_max, captures "
                      "FROM psd_meta WHERE sensor=?", [sensor]).fetchone()
        rep.check(m is not None, "psd_meta row exists for this sensor")
        if m:
            rep.check(m[0] == F0 and m[1] == DF and m[2] == NF,
                      "psd_meta geometry", f"f0={m[0]:.0f} df={m[1]:.0f} nf={m[2]}")
            rep.check(m[3] == QMIN and m[4] == QMAX, "psd_meta quantization range",
                      f"[{m[3]}, {m[4]}]")
            rep.check(m[7] == len(dts), "psd_meta capture count matches stored",
                      f"meta {m[7]:,} vs stored {len(dts):,}")
            if strict_counts:
                rep.check(m[7] == n, "psd_meta capture count matches the csv",
                          f"meta {m[7]:,} vs csv {n:,}")
            first, last = float(stamps[0]) / 1e6, float(stamps[-1]) / 1e6
            ok_span = (m[5] is not None and m[6] is not None
                       and m[5] <= first and last <= m[6] + 1e-3)
            rep.check(ok_span, "psd_meta time span covers this csv end to end",
                      f"meta [{m[5]}, {m[6]}] vs csv [{first}, {last}]")

        # -- CHECK 6: the coarse pyramid. Recomputed from the CSV floats, not
        #    from the database's fine table -- a pyramid that agrees with wrong
        #    captures is still wrong.
        tabs = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        if "psd_lvl" in tabs:
            lv = c.execute("SELECT bucket, t, smax FROM psd_lvl WHERE sensor=? "
                           "ORDER BY bucket, t", [sensor]).fetchall()
            span_lo, span_hi = float(dts.min()), float(dts.max())
            stray = [(b, t) for b, t, _s in lv
                     if not (span_lo - b <= t <= span_hi)]
            rep.check(not stray, "no psd_lvl row outside the stored capture span",
                      f"{len(lv)} row(s), {len(stray)} stray"
                      + (f"; e.g. bucket {stray[0][0]} at t={stray[0][1]}"
                         if stray else ""))
            csv_us = set(int(v) for v in want_us)
            # want bytes derived from the CSV, independent of `got`
            ref_csv = ref
            overlap = checked = wrong = 0
            for bucket, t, blob in lv:
                sel = np.flatnonzero((dts >= t) & (dts < t + bucket))
                if not sel.size:
                    continue
                overlap += 1
                if any(int(v) not in csv_us and int(v) - 1 not in csv_us
                       and int(v) + 1 not in csv_us for v in have_us[sel]):
                    continue                 # bucket mixes in other days
                rows_in = []
                for v in have_us[sel]:
                    hit = np.flatnonzero(np.abs(want_us - int(v)) <= 1)
                    if hit.size:
                        rows_in.append(int(hit[0]))
                if len(rows_in) != sel.size:
                    continue
                want = ref_csv[rows_in].max(axis=0)
                have = np.frombuffer(blob, dtype=np.uint8)
                checked += 1
                if have.size != NF or not np.array_equal(want, have):
                    wrong += 1
            # A pyramid table that exists but has nothing checkable is a failure,
            # not a pass: an empty psd_lvl means wide windows draw nothing.
            rep.check(wrong == 0 and checked > 0,
                      "psd_lvl rows are the bin-wise max of the csv",
                      f"{checked} of {overlap} overlapping bucket(s) predictable "
                      f"from this csv, {wrong} wrong"
                      + ("  [NOTHING CHECKABLE -- empty or foreign pyramid]"
                         if not checked else ""))
        else:
            rep.check(True, "psd_lvl absent",
                      "no coarse pyramid built; wide windows use the capture path")
    finally:
        c.close()
    return rep, picks


# --------------------------------------------------------------- build mode

def build_isolated(csv_path, sensor, stat, workdir, do_compact=True,
                   do_levels=True, root=None):
    """Ingest into a fresh database and return its path.

    `root` runs the ingest against the WHOLE export directory rather than a
    private copy of the files being checked. That matters: pointing it at a
    private copy means the ingest's own file selection -- which sensor, which
    statistic -- is never asked to choose, so an ingest that fed max data into
    the mean database would pass. Falls back to a private copy when no root is
    given, for the single-file case.
    """
    src = root
    if src is None:
        paths = [csv_path] if isinstance(csv_path, str) else list(csv_path)
        src = os.path.join(workdir, "src")
        os.makedirs(src, exist_ok=True)
        for p in paths:
            shutil.copy2(p, os.path.join(src, os.path.basename(p)))
    env = dict(os.environ)
    env["ATLAS_DB_DIR"] = workdir
    env.pop("PSD_DB", None)
    steps = [([sys.executable, os.path.join(ROOT, "ingest", "psd_ingest.py"),
               sensor, "--stat", stat, "--root", src], "ingest")]
    if do_compact:
        steps.append(([sys.executable, os.path.join(ROOT, "ingest", "compact_db.py")],
                      "compact"))
    if do_levels:
        steps.append(([sys.executable, os.path.join(ROOT, "ingest",
                                                    "build_psd_levels.py"),
                       "--stat", stat], "levels"))
    log = []
    for cmd, name in steps:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        log.append(f"--- {name} (exit {p.returncode}) ---\n{p.stdout}{p.stderr}")
        if p.returncode != 0:
            raise RuntimeError(f"{name} failed for {sensor}/{stat}:\n" + log[-1])
    name = "psd.duckdb" if stat == "max" else f"psd_{stat}.duckdb"
    return os.path.join(workdir, name), "\n".join(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="the export to verify against")
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--stat", required=True, choices=["max", "median", "mean"])
    ap.add_argument("--db", default=None,
                    help="verify this existing database (read-only). "
                         "Omit to build an isolated one.")
    ap.add_argument("--root", default=None,
                    help="in build mode, run the ingest against this whole "
                         "export directory (exercises its own file selection)")
    ap.add_argument("--cross", nargs="*", default=[],
                    help="the other statistics' CSVs; the database must NOT "
                         "match these")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--bytes", type=int, default=20000, dest="nbytes",
                    help="cells cross-checked with exact rational arithmetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many sample points as a table")
    ap.add_argument("--strict-counts", action="store_true",
                    help="the database must hold EXACTLY the csv's captures "
                         "(only true when it was built from just these files)")
    ap.add_argument("--keep", action="store_true", help="keep the build dir")
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only failures")
    args = ap.parse_args()

    if args.nbytes <= 0 or args.samples < 0:
        print("--bytes must be positive and --samples non-negative: a zero "
              "budget would disable a check and still report ok", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    label = f"{args.sensor} / {args.stat} / {os.path.basename(args.csv)}"
    tmp = None
    strict = args.strict_counts or (args.db is None and args.root is None)
    try:
        if args.db:
            dbpath = args.db
        else:
            tmp = tempfile.mkdtemp(prefix="verify1to1-")
            dbpath, _log = build_isolated(args.csv, args.sensor, args.stat, tmp,
                                          do_compact=not args.no_compact,
                                          root=args.root)
        rep, picks = verify(args.csv, args.sensor, args.stat, dbpath, rng,
                            args.samples, args.nbytes, label,
                            cross=args.cross, strict_counts=strict)
    finally:
        if tmp and not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print(rep.text(verbose=not args.quiet))
    if args.show and picks:
        print(f"    {'capture (utc)':26} {'freq MHz':>11} {'csv dBm/Hz':>11} "
              f"{'byte':>5} {'stored':>11} {'delta':>9}")
        for i, j, v, b, d, f, us in picks[:args.show]:
            print(f"    {np.datetime64(us, 'us')!s:26} {f/1e6:11.3f} {v:11.2f} "
                  f"{b:5d} {d:11.4f} {d - v:+9.4f}")
    return 1 if rep.bad else 0


if __name__ == "__main__":
    sys.exit(main())
