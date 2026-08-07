"""verify_all_stats.py: run the 1:1 CSV check over every sensor and statistic.

Discovers whatever <YYYY-MM-DD>_<sensor>_<stat>.csv files are under --root,
builds one isolated database per (sensor, statistic) -- ingest, compact, coarse
levels, the whole path a real install takes, run against the whole export
directory so the ingest's own file selection is exercised -- and verifies every
CSV that went into it against what came out. Nothing existing is touched; each
combination gets its own temporary directory.

    py examples/verify_all_stats.py --root /path/to/SEA-DATA --expect 30
    py examples/verify_all_stats.py --root ... --jobs 8

`--db-dir` switches it to checking databases that already exist (read-only)
instead of building fresh ones -- that is the mode for an install you want to
audit rather than reproduce.

`--expect N` is worth using. Without it the total is self-reported by the same
discovery pass that produced the checks, so a mistyped `--sensors` prints a
clean "0/0 verified" and exits 0.
"""

import argparse
import concurrent.futures as cf
import gzip
import os
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import verify_csv_1to1 as V                              # noqa: E402

NAME_RE = re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})_(?P<sensor>.+?)_"
                     r"(?P<stat>[A-Za-z0-9]+)\.csv$")
STATS = ("max", "median", "mean")


def discover(root):
    """{(sensor, stat): {day: path}} for every PSD export under root."""
    found = {}
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            m = NAME_RE.match(n)
            if not m or m.group("stat") not in STATS:
                continue
            found.setdefault((m.group("sensor"), m.group("stat")), {})[
                m.group("day")] = os.path.join(dirpath, n)
    return found


def csv_rows(path):
    """How many data rows an export holds, without parsing its values."""
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rb") as fh:
        return max(sum(1 for line in fh if line.strip()) - 1, 0)


def one(sensor, stat, paths, all_paths, cross, root, db_dir, samples, nbytes,
        seed, keep):
    """Verify one (sensor, statistic). Returns (label, ok, text, seconds)."""
    t0 = time.time()
    label = f"{sensor:22} {stat:6}"
    tmp = None
    try:
        if db_dir:
            name = "psd.duckdb" if stat == "max" else f"psd_{stat}.duckdb"
            dbpath = os.path.join(db_dir, name)
            if not os.path.exists(dbpath):
                return label, False, f"    FAIL no database at {dbpath}", 0.0
        else:
            tmp = tempfile.mkdtemp(prefix=f"v30-{stat}-")
            dbpath, _log = V.build_isolated(paths, sensor, stat, tmp, root=root)
        blocks, ok = [], True
        for i, p in enumerate(paths):
            rng = np.random.default_rng(seed + i)
            rep, _picks = V.verify(p, sensor, stat, dbpath, rng, samples,
                                   nbytes, f"{sensor} / {stat} / "
                                   f"{os.path.basename(p)}",
                                   cross=cross.get(os.path.basename(p), ()))
            ok = ok and not rep.bad
            blocks.append(rep.text(verbose=True))

        if not db_dir:
            # Built from exactly these files, so the database must hold exactly
            # their captures -- containment alone would let a duplicated chunk or
            # a re-ingest through. Counted over EVERY day of the combination, not
            # the subset being verified: with --root the ingest reads all of them,
            # and comparing against a --days-capped subset failed on correct data.
            total_rows = sum(csv_rows(p) for p in all_paths)
            import duckdb
            c = duckdb.connect(dbpath, read_only=True)
            try:
                kind = V.db_kind(c)
                dts, _sp = V.db_read(c, sensor, kind)
                meta = c.execute("SELECT captures FROM psd_meta WHERE sensor=?",
                                 [sensor]).fetchone()
            finally:
                c.close()
            exact = len(dts) == total_rows and meta and meta[0] == total_rows
            ok = ok and bool(exact)
            blocks.append(
                f"{'PASS' if exact else 'FAIL'}  {sensor} / {stat} / totals\n"
                f"    {'ok  ' if exact else 'FAIL'} database holds exactly the "
                f"{len(all_paths)} export(s)' captures  -- stored {len(dts):,}, "
                f"meta {meta[0] if meta else 'none'}, csv total {total_rows:,}")
        return label, ok, "\n".join(blocks), time.time() - t0
    except Exception:
        return label, False, "    FAIL exception\n" + traceback.format_exc(), \
            time.time() - t0
    finally:
        if tmp and not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("SEA_DATA_ROOT")
                    or os.path.join(ROOT, "SEA-DATA"))
    ap.add_argument("--db-dir", default=None,
                    help="verify existing databases in this directory instead "
                         "of building fresh ones")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--days", type=int, default=None,
                    help="cap how many days per combination (default: all found)")
    ap.add_argument("--sensors", nargs="*", default=None)
    ap.add_argument("--stats", nargs="*", default=None, choices=list(STATS))
    ap.add_argument("--expect", type=int, default=None,
                    help="fail unless exactly this many combinations were run")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--bytes", type=int, default=20000, dest="nbytes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--detail", action="store_true",
                    help="print every check, not just the verdict line")
    args = ap.parse_args()

    if args.nbytes <= 0:
        print("--bytes must be positive", file=sys.stderr)
        return 2

    found = discover(args.root)
    if not found:
        print(f"no PSD exports found under {os.path.abspath(args.root)}",
              file=sys.stderr)
        return 2
    sensors = sorted({s for s, _ in found})
    if args.sensors:
        unknown = [s for s in args.sensors if s not in sensors]
        if unknown:
            print(f"no exports for sensor(s): {', '.join(unknown)}\n"
                  f"  on disk: {', '.join(sensors)}", file=sys.stderr)
            return 2
        sensors = [s for s in sensors if s in args.sensors]
    stats = list(args.stats or STATS)

    combos, skipped = [], []
    for s in sensors:
        for st in stats:
            days = found.get((s, st))
            if not days:
                skipped.append(f"{s}/{st}")
                continue
            all_paths = [days[d] for d in sorted(days)]
            paths = all_paths[:args.days] if args.days else all_paths
            # the same days of the OTHER statistics: this database must not match
            cross = {}
            for p in paths:
                day = os.path.basename(p).split("_")[0]
                cross[os.path.basename(p)] = tuple(
                    found[(s, o)][day] for o in STATS
                    if o != st and (s, o) in found and day in found[(s, o)])
            combos.append((s, st, paths, all_paths, cross))

    if not combos:
        print("nothing to verify: no (sensor, statistic) combination matched",
              file=sys.stderr)
        return 2

    nfiles = sum(len(p) for _s, _t, p, _a, _c in combos)
    print(f"{len(combos)} combination(s) over {len(sensors)} sensor(s), "
          f"{nfiles} export file(s), {args.jobs} parallel job(s)"
          + (f", against existing databases in {args.db_dir}" if args.db_dir
             else ", each built from scratch")
          + (f"  [--days {args.days}: only the first {args.days} day(s) per "
             f"combination]" if args.days else ""))
    if skipped:
        print("  no exports on disk for: " + ", ".join(skipped))
    print()

    t0 = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, s, st, p, a, c,
                          None if args.db_dir else args.root, args.db_dir,
                          args.samples, args.nbytes, args.seed,
                          args.keep): (s, st)
                for s, st, p, a, c in combos}
        for fut in cf.as_completed(futs):
            label, ok, text, secs = fut.result()
            results.append((label, ok, text, secs))
            print(f"{'PASS' if ok else 'FAIL'}  {label} {secs:6.1f}s", flush=True)
            if not ok or args.detail:
                print(text, flush=True)

    results.sort(key=lambda r: r[0])
    bad = [r for r in results if not r[1]]
    print("\n" + "=" * 72)
    for label, ok, _text, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {secs:6.1f}s")
    print("=" * 72)
    print(f"{len(results) - len(bad)}/{len(results)} combination(s) verified 1:1 "
          f"over {nfiles} export file(s) in {(time.time() - t0) / 60:.1f} min")
    shortfall = (args.expect is not None and len(results) != args.expect)
    if shortfall:
        print(f"EXPECTED {args.expect} combination(s), ran {len(results)}",
              file=sys.stderr)
    if bad:
        print("\nFAILED:")
        for label, _ok, text, _s in bad:
            print(f"\n--- {label} ---\n{text}")
    return 1 if (bad or shortfall) else 0


if __name__ == "__main__":
    sys.exit(main())
