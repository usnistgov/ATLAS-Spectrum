"""test_verify_1to1.py: prove verify_csv_1to1.py can actually fail.

A verifier that passes on corrupted data is worse than none, because it is
quoted as evidence. So build one real database from a real (or fixture) CSV,
then damage it in the specific ways the ingest could plausibly get wrong, and
require the verifier to catch each one -- and to name a check that plausibly
owns the damage, not merely to exit non-zero for some unrelated reason.

    py examples/test_verify_1to1.py                       # own fixture
    py examples/test_verify_1to1.py --csv <file>.csv --sensor HU --stat max \
        --cross <same-day-other-stat>.csv       # adds the wrong-statistic case

Two kinds of case, and the distinction matters: MUST_CATCH must make the verifier
FAIL, and MUST_REPORT must leave it passing while still saying something (a
missing coarse pyramid is a legitimate install, not damage). The counts are
printed at the end rather than written here, so they cannot drift.

The cases that matter most are the ones an earlier version of the verifier
missed: a round-half-up quantizer (invisible to a half-step tolerance), a
database wired to the wrong statistic, an emptied or fabricated coarse pyramid,
duplicated captures with the metadata updated to agree, and a collapsed time
span. Each of those passed cleanly once.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import duckdb                                            # noqa: E402
import verify_csv_1to1 as V                              # noqa: E402

NF = V.NF


def make_fixture(path, sensor, n=40, seed=7, ties=True):
    """A small export. `ties` sprinkles in the values that land exactly halfway
    between two bytes, which is what the tie-rule check needs to bite."""
    rng = np.random.default_rng(seed)
    freqs = V.F0 + np.arange(NF) * V.DF
    base = 1752019339.307
    with open(path, "w", newline="") as fh:
        fh.write("sweep_timestamp," + ",".join(f"{f:.1f}" for f in freqs) + "\n")
        for i in range(n):
            t = base + i * 120.0
            us = int(round(t * 1e6))
            s = str(np.datetime64(us, "us")).replace("T", " ") + "+00:00"
            v = -150 + 30 * rng.random(NF)
            v = np.round(v, 1)
            if ties:
                where = rng.choice(NF, size=NF // 20, replace=False)
                v[where] = rng.choice(V.TIE_VALUES, size=where.size)
            fh.write(s + "," + ",".join(f"{x:.1f}" for x in v) + "\n")
    return path


def run_verify(csv, sensor, stat, db, extra=()):
    p = subprocess.run([sys.executable, os.path.join(HERE, "verify_csv_1to1.py"),
                        "--csv", csv, "--sensor", sensor, "--stat", stat,
                        "--db", db, "--samples", "40", "--bytes", "4000",
                        "--strict-counts", *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def failed_checks(out):
    """The named checks that failed, ignoring the report's own header line."""
    return [ln.strip()[5:].strip() for ln in out.splitlines()
            if ln.startswith("    FAIL ")]


def _chunk(c):
    rows = c.execute("SELECT rowid, sensor, t0, times, specs, n FROM psd_chunk "
                     "ORDER BY t0").fetchall()
    if not rows:
        raise RuntimeError("no chunks to mutate")
    rid, sensor, t0, tb, sb, n = rows[0]
    ts = np.frombuffer(zlib.decompress(tb), dtype=np.float64).copy()
    sp = np.frombuffer(zlib.decompress(sb), dtype=np.uint8).copy().reshape(len(ts), NF)
    return rid, sensor, ts, sp, n


def _write(c, rid, ts, sp, n):
    c.execute("UPDATE psd_chunk SET times=?, specs=?, n=? WHERE rowid=?",
              [zlib.compress(ts.tobytes(), 9),
               zlib.compress(sp.tobytes(), 9), int(n), rid])


def mutate(db, how, csv):
    """Damage a copy of the database in one specific way. Returns a note."""
    c = duckdb.connect(db)
    try:
        if how == "half_up_at_ties":
            # A round-half-up quantizer differs from round-half-to-even ONLY at
            # the halfway values, and both bytes are within half a step there --
            # so nothing but the tie check can see this.
            stamps, _f, vals = V.read_csv_independently(csv)
            rid, _s, ts, sp, n = _chunk(c)
            tie = np.isfinite(vals) & V.is_tie(vals)
            hit = 0
            for i in range(min(len(ts), len(stamps))):
                for j in np.flatnonzero(tie[i]):
                    x = (float(vals[i, j]) + 180.0) / 90.0 * 255.0
                    sp[i, j] = int(np.floor(x)) + 1      # always up
                    hit += 1
            _write(c, rid, ts, sp, n)
            return f"{hit} halfway cells rounded half-UP instead of half-to-even"
        if how == "empty_levels":
            c.execute("DELETE FROM psd_lvl")
            return "psd_lvl emptied"
        if how == "drop_levels":
            c.execute("DROP TABLE IF EXISTS psd_lvl")
            return "psd_lvl dropped"
        if how == "fabricated_levels":
            row = c.execute("SELECT sensor, bucket, t FROM psd_lvl "
                            "ORDER BY bucket, t LIMIT 1").fetchone()
            if not row:
                return None
            c.execute("DELETE FROM psd_lvl")
            for k in range(20):
                c.execute("INSERT INTO psd_lvl VALUES (?,?,?,?)",
                          [row[0], row[1], float(row[2]) + 86400 * 400 + k * row[1],
                           bytes([255]) * NF])
            return "psd_lvl replaced with 20 fabricated all-255 rows a year later"
        if how == "duplicate_chunk_and_meta":
            rid, _s, _ts, _sp, _n = _chunk(c)
            rows = c.execute("SELECT sensor, t0, t1, n, times, specs FROM "
                             "psd_chunk").fetchall()
            for sensor, t0, t1, n, tb, sb in rows:
                ts = np.frombuffer(zlib.decompress(tb), dtype=np.float64) + 0.4
                c.execute("INSERT INTO psd_chunk (sensor, t0, t1, n, times, specs) "
                          "VALUES (?,?,?,?,?,?)",
                          [sensor, float(ts[0]), float(ts[-1]), n,
                           zlib.compress(ts.tobytes(), 9), sb])
            c.execute("UPDATE psd_meta SET captures = captures * 2")
            return "every capture duplicated 400 ms later, meta updated to agree"
        if how == "collapse_t_max":
            c.execute("UPDATE psd_meta SET t_max = t_min")
            return "psd_meta.t_max collapsed onto t_min"
        if how == "shift_one_bin":
            rid, _s, ts, sp, n = _chunk(c)
            _write(c, rid, ts, np.roll(sp, 1, axis=1), n)
            return "first chunk's bins rolled by one"
        if how == "off_by_one_count":
            rid, _s, ts, sp, n = _chunk(c)
            sp[3, 100] = (int(sp[3, 100]) + 1) % 256
            _write(c, rid, ts, sp, n)
            return "one byte changed by one count"
        if how == "one_count_offset":
            rid, _s, ts, sp, n = _chunk(c)
            _write(c, rid, ts, np.clip(sp.astype(np.int16) + 1, 0, 255)
                   .astype(np.uint8), n)
            return "first chunk shifted up one count (0.353 dB)"
        if how == "reverse_bins":
            rid, _s, ts, sp, n = _chunk(c)
            _write(c, rid, ts, sp[:, ::-1], n)
            return "frequency axis reversed"
        if how == "swap_two_captures":
            rid, _s, ts, sp, n = _chunk(c)
            sp[[0, 1]] = sp[[1, 0]]
            _write(c, rid, ts, sp, n)
            return "two captures' spectra swapped"
        if how == "drop_a_capture":
            rid, _s, ts, sp, n = _chunk(c)
            _write(c, rid, ts[1:], sp[1:], len(ts) - 1)
            return "one capture removed"
        if how == "nudge_timestamp":
            rid, _s, ts, sp, n = _chunk(c)
            ts[2] += 0.010
            _write(c, rid, ts, sp, n)
            return "one timestamp moved 10 ms"
        if how == "nudge_timestamp_us":
            rid, _s, ts, sp, n = _chunk(c)
            ts[2] += 0.000400
            _write(c, rid, ts, sp, n)
            return "one timestamp moved 400 us"
        if how == "duplicate_chunk":
            rid, _s, _ts, _sp, _n = _chunk(c)
            c.execute("INSERT INTO psd_chunk (sensor, t0, t1, n, times, specs) "
                      "SELECT sensor, t0, t1, n, times, specs FROM psd_chunk "
                      "WHERE rowid=?", [rid])
            return "second copy of the first chunk"
        if how == "break_meta_count":
            c.execute("UPDATE psd_meta SET captures = captures + 5")
            return "psd_meta.captures inflated by 5"
        if how == "break_meta_geometry":
            c.execute("UPDATE psd_meta SET df = 100000")
            return "psd_meta.df changed to 100 kHz"
        if how == "drop_meta":
            c.execute("DELETE FROM psd_meta")
            return "psd_meta row removed"
        if how == "break_level":
            lv = c.execute("SELECT rowid, smax FROM psd_lvl "
                           "WHERE bucket=(SELECT min(bucket) FROM psd_lvl) "
                           "ORDER BY t").fetchall()
            if not lv:
                return None
            rid2, blob = lv[len(lv) // 2]
            b = bytearray(blob)
            b[7] = (b[7] + 3) % 256
            c.execute("UPDATE psd_lvl SET smax=? WHERE rowid=?", [bytes(b), rid2])
            return "one psd_lvl byte changed"
        if how == "break_all_levels":
            lv = c.execute("SELECT rowid, smax FROM psd_lvl").fetchall()
            if not lv:
                return None
            for rid2, blob in lv:
                b = bytearray(blob)
                b[11] = (b[11] + 7) % 256
                c.execute("UPDATE psd_lvl SET smax=? WHERE rowid=?",
                          [bytes(b), rid2])
            return f"{len(lv)} psd_lvl rows damaged"
        raise ValueError(how)
    finally:
        c.close()


# Cases that must FAIL the verifier, and the check that should own each.
MUST_CATCH = {
    "half_up_at_ties": "halfway values",
    "empty_levels": "psd_lvl rows",
    "fabricated_levels": "psd_lvl",
    "duplicate_chunk_and_meta": "exactly the csv's captures",
    "collapse_t_max": "psd_meta time span",
    "shift_one_bin": "round-trips within half a step",
    "off_by_one_count": "round-trips within half a step",
    "one_count_offset": "round-trips within half a step",
    "reverse_bins": "round-trips within half a step",
    "swap_two_captures": "round-trips within half a step",
    "drop_a_capture": "capture is stored",
    "nudge_timestamp": "capture is stored",
    "nudge_timestamp_us": "capture is stored",
    "duplicate_chunk": "duplicate capture instants",
    "break_meta_count": "psd_meta capture count",
    "break_meta_geometry": "psd_meta geometry",
    "drop_meta": "psd_meta row exists",
    "break_level": "psd_lvl rows",
    "break_all_levels": "psd_lvl rows",
}
# Cases that must PASS but must SAY something, because they are legitimate states
# rather than damage.
MUST_REPORT = {"drop_levels": "psd_lvl absent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sensor", default="FIXTURE")
    ap.add_argument("--stat", default="max", choices=["max", "median", "mean"])
    ap.add_argument("--cross", default=None,
                    help="another statistic's CSV for the same sensor and day; "
                         "enables the wrong-statistic routing case")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="testverify-")
    try:
        csv = args.csv
        if csv is None:
            csv = os.path.join(tmp, f"2025-07-09_{args.sensor}_{args.stat}.csv")
            make_fixture(csv, args.sensor)
        good = os.path.join(tmp, "build")
        os.makedirs(good)
        db, _ = V.build_isolated(csv, args.sensor, args.stat, good)
        rc, out = run_verify(csv, args.sensor, args.stat, db)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} clean database verifies "
              f"(exit {rc})")
        if rc != 0:
            print(out)
            return 1

        bad = 0
        for how, owns in MUST_CATCH.items():
            work = os.path.join(tmp, "m_" + how)
            os.makedirs(work, exist_ok=True)
            dup = os.path.join(work, os.path.basename(db))
            shutil.copy2(db, dup)
            note = mutate(dup, how, csv)
            if note is None:
                print(f"FAIL {how:26} -> nothing to damage (case is vacuous)")
                bad += 1
                continue
            rc, out = run_verify(csv, args.sensor, args.stat, dup)
            names = failed_checks(out)
            owned = any(owns in nm for nm in names)
            ok = rc != 0 and owned
            print(f"{'ok  ' if ok else 'FAIL'} {how:26} -> "
                  + (f"caught by: {names[0][:70]}" if names else "NOT CAUGHT")
                  + ("" if owned else f"   [expected a check about '{owns}']"))
            if not ok:
                bad += 1

        for how, says in MUST_REPORT.items():
            work = os.path.join(tmp, "r_" + how)
            os.makedirs(work, exist_ok=True)
            dup = os.path.join(work, os.path.basename(db))
            shutil.copy2(db, dup)
            mutate(dup, how, csv)
            rc, out = run_verify(csv, args.sensor, args.stat, dup)
            ok = rc == 0 and says in out
            print(f"{'ok  ' if ok else 'FAIL'} {how:26} -> "
                  f"passes and reports '{says}'" if ok else
                  f"FAIL {how:26} -> exit {rc}, '{says}' not reported")
            if not ok:
                bad += 1

        # -- wrong statistic wired to the wrong database. Needs a second CSV.
        if args.cross:
            rc, out = run_verify(args.cross, args.sensor, args.stat, db)
            caught = rc != 0
            print(f"{'ok  ' if caught else 'FAIL'} {'wrong_stat_csv':26} -> "
                  + (f"caught by: {failed_checks(out)[0][:70]}" if caught
                     else "NOT CAUGHT"))
            if not caught:
                bad += 1
            rc, out = run_verify(csv, args.sensor, args.stat, db,
                                 extra=("--cross", args.cross))
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {'cross-check on good db':26}"
                  f" -> exit {rc} (must stay 0)")
            if rc != 0:
                bad += 1

        # -- a blank value cell has no correct byte; the CSV side must refuse it.
        with open(csv) as fh:
            lines = fh.read().splitlines()
        parts = lines[2].split(",")
        parts[500] = ""
        lines[2] = ",".join(parts)
        blank = os.path.join(tmp, "blank.csv")
        with open(blank, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        rc, out = run_verify(blank, args.sensor, args.stat, db)
        ok = rc != 0 and any("blank / NaN" in nm for nm in failed_checks(out))
        print(f"{'ok  ' if ok else 'FAIL'} {'blank_value_cell':26} -> "
              + ("caught by the NaN policy" if ok else "NOT CAUGHT"))
        if not ok:
            bad += 1

        total = len(MUST_CATCH) + len(MUST_REPORT) + 1 + (2 if args.cross else 0)
        print(f"\n{len(MUST_CATCH)} damage case(s) that must be caught, "
              f"{len(MUST_REPORT)} legitimate state(s) that must still pass, "
              f"{total} case(s) run")
        print('ALL CASES BEHAVED' if not bad else f'{bad} CASE(S) FAILED')
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
