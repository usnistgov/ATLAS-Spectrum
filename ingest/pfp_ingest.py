"""
pfp_ingest.py: ingest PFP (periodic-frame-power) numbers into a compact store.

PFP = power across a 10 ms frame (560 positions, ~17.86 us each) per 10 MHz
channel, one trace per sweep (the SEA schedule sweeps every ~90 s, dwelling 4 s
on each of the 18 channels). We store one uint8-quantized BLOB per
(channel, capture): (sensor, freq, t, frame). Re-render any channel/time window
crisply, like the PSD layer.

Source CSVs are found by walking --root (default $SEA_DATA_ROOT, else
./SEA-DATA) and matching PFP_<YYYY-MM-DD>_<sensor>_<stat>.csv anywhere
underneath, so the folder layout does not matter.

The server reads this file directly. compact_db.py repacks it into a smaller
chunked form and swaps that in; that step is now optional and only affects
size and speed, not whether the viewer works. Re-running against a database
that is already compacted appends in the chunked shape rather than the row
shape, so next month's data lands where the server will read it.

    py pfp_ingest.py --list                          # what is on disk
    py pfp_ingest.py CBBT-Directional                # default stat max_peak
    py pfp_ingest.py CBBT-Directional --limit 30     # quick prototype
    py pfp_ingest.py CBBT-Directional --stat mean_rms
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cbrs_files                                    # noqa: E402
import chunk_io                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root (scripts live in ingest/)
# Where the CBRS source CSVs live. Set SEA_DATA_ROOT to your copy, or pass
# --root; defaults to ./SEA-DATA in the repo so a fresh clone never points at
# someone else's disk.
DATA_ROOT = os.environ.get("SEA_DATA_ROOT", os.path.join(ROOT, "SEA-DATA"))
# abspath for the same reason serve.py does it: a relative ATLAS_DB_DIR resolves
# against each step's cwd, so the ingest and compact_db.py silently worked on
# two different files.
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
PFP_DB = os.environ.get("PFP_DB") or os.path.join(DB_DIR, "pfp.duckdb")

NPOS = 560
FRAME_MS = 10.0
QMIN, QMAX = -130.0, -10.0   # uint8 range (dBm), ~0.47 dB/step

# PFP_<YYYY-MM-DD>_<sensor>_<stat>.csv , where stat may itself contain one
# underscore (max_peak, mean_rms). The lazy sensor match resolves the rest.
NAME_RE = re.compile(r"^PFP_(?P<day>\d{4}-\d{2}-\d{2})_(?P<sensor>.+?)_"
                     r"(?P<stat>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)\.csv$")
PATTERN = "PFP_<YYYY-MM-DD>_<sensor>_<stat>.csv"

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def discover(root, stat=None, collisions=None):
    return cbrs_files.discover(root, NAME_RE, stat, collisions)


def _rollback(connection, apps):
    return chunk_io.rollback_day(connection, apps)


def stats_present(root):
    return cbrs_files.stats_present(root, NAME_RE)


def read_frames(path, csv_connection):
    """CSV -> (epoch_seconds[n], freq_hz[n], quantized uint8 [n,560])."""
    ts, keys, Q = cbrs_files.read_quantized(
        path, csv_connection, NPOS, QMIN, QMAX, "dBm", "frame-position",
        lead=("frequency",))
    return ts, keys["frequency"], Q


def main():
    ap = argparse.ArgumentParser(
        description="Ingest CBRS PFP CSVs into pfp.duckdb.")
    ap.add_argument("sensor", nargs="?", default=None,
                    help="sensor name; omit when only one is on disk")
    ap.add_argument("--root", default=DATA_ROOT,
                    help=f"directory holding the source CSVs (default {DATA_ROOT})")
    ap.add_argument("--stat", default="max_peak",
                    help="which statistic to ingest (default max_peak)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N days, for a quick test")
    ap.add_argument("--list", action="store_true",
                    help="show the sensors and day counts found, then exit")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        sys.exit(cbrs_files.missing_root(root))

    collisions = {}
    found = discover(root, args.stat, collisions)
    cbrs_files.report_collisions(collisions, "sensor-day")
    if not found:
        sys.exit(cbrs_files.no_data(root, args.stat, NAME_RE,
                                    PATTERN, "PFP"))

    if args.list:
        print(f"{os.path.abspath(root)} (stat: {args.stat})\n")
        for s in sorted(found):
            days = sorted(found[s])
            print(f"  {s:24} {len(days):>5} day(s)  {days[0]} .. {days[-1]}")
        print(f"\nstats available: {', '.join(stats_present(root))}")
        print("\nnext: python ingest/pfp_ingest.py <sensor>")
        return 0

    sensor, why = cbrs_files.resolve_sensor(args.sensor, found, root)
    if why:
        sys.exit(why)
    if args.sensor is None:
        print(f"one sensor on disk; ingesting {sensor}")

    by_day = found[sensor]
    days = sorted(by_day)
    print(f"{sensor} PFP ({args.stat}): {len(days)} day(s) on disk")

    connection = cbrs_files.open_db(PFP_DB, duckdb)
    connection.execute("CREATE TABLE IF NOT EXISTS pfp (sensor VARCHAR, freq DOUBLE, t DOUBLE, frame BLOB)")
    connection.execute("""CREATE TABLE IF NOT EXISTS pfp_meta (
        sensor VARCHAR, stat VARCHAR, npos INT, frame_ms DOUBLE, qmin DOUBLE, qmax DOUBLE,
        t_min DOUBLE, t_max DOUBLE, rows BIGINT)""")
    # This database may already be compacted -- see the note in psd_ingest.py.
    # Append in whichever shape is on disk, so a second run months later lands
    # where serve.py will actually read it.
    kind = chunk_io.schema_of(connection, "pfp")
    if kind == "chunk":
        print("  this database is compacted; appending new days as chunks")
    # Resumable, by exactly the rule psd_ingest.py uses -- see
    # chunk_io.day_is_ingested. Count the days THIS RUN skips rather than every
    # day the sensor has stored: the table spans days that are not under this
    # --root at all, and counting those is how "1 day(s) on disk / 2 day(s)
    # already ingested" came about.
    times = chunk_io.stored_times(connection, "pfp", sensor, kind)
    stored_ms = {int(round(t * 1000)) for t in times}
    per_day = chunk_io.captures_per_day(times)

    def have(d):
        return chunk_io.day_is_ingested(
            d, per_day, stored_ms, lambda: cbrs_files.first_capture_time(by_day[d]))

    skipping = sum(1 for d in days if have(d))
    if skipping:
        print(f"  resuming: {skipping} of {len(days)} day(s) already "
              f"ingested, skipping those")
    csv_connection = duckdb.connect()

    t0 = time.time()
    done = err = nrows = 0
    errors = []
    # A chunk is one channel's consecutive frames, so the appenders are keyed by
    # frequency. They are kept ACROSS days but flushed at the end of each one --
    # see below. Keeping them open across days without flushing packed fuller
    # chunks, and cost atomicity to do it.
    apps = {}
    # One transaction per day, and the appenders are flushed inside it. DuckDB
    # autocommits per parameter set, so an interrupted day used to leave an
    # arbitrary prefix committed -- and `have()` then counts those frames, calls
    # the day done, and skips the rest forever. Flushing per day costs slightly
    # shorter chunks (compact_db.py repacks them) and buys atomicity: a day that
    # did not finish contributes nothing, so "was it started" and "was it
    # finished" become the same question.
    interrupts = tuple(x for x in (KeyboardInterrupt,
                                   getattr(duckdb, "InterruptException", None))
                       if x is not None)
    for i, day in enumerate(days, 1):
        if have(day):
            done += 1
            continue
        path = by_day[day]
        try:
            connection.execute("BEGIN TRANSACTION")
            ts, freq, Q = read_frames(path, csv_connection)
            if not len(ts):
                raise ValueError("the file has a header but no data rows")
            if kind == "chunk":
                for f in np.unique(np.asarray(freq, dtype=np.float64)):
                    sel = np.flatnonzero(np.asarray(freq, dtype=np.float64) == f)
                    app = apps.get(float(f))
                    if app is None:
                        app = apps[float(f)] = chunk_io.ChunkAppender(
                            connection, "pfp", sensor, "frames", key=float(f))
                    app.add([ts[j] for j in sel], [Q[j] for j in sel])
            else:
                connection.executemany(
                    "INSERT INTO pfp VALUES (?,?,?,?)",
                    [(sensor, float(freq[j]), float(ts[j]), Q[j].tobytes())
                     for j in range(len(ts))])
            for app in apps.values():
                app.flush()          # inside this day's transaction
            connection.execute("COMMIT")
            done += 1; nrows += len(ts)
        except interrupts:
            _rollback(connection, apps)
            print(f"\ninterrupted during {os.path.basename(path)}; that day was "
                  f"rolled back. Re-run to continue where this left off.",
                  file=sys.stderr)
            connection.close()
            return 130
        except Exception as e:
            _rollback(connection, apps)
            err += 1
            errors.append(f"{os.path.basename(path)}: {e}")
            print(f"  ERR {os.path.basename(path)}: {e}")
        if i % 10 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] rows={nrows:,} done={done} err={err} "
                  f"{i/max(time.time()-t0,1e-6):.2f} days/s")

    r = chunk_io.stored_span(connection, "pfp", sensor, chunk_io.schema_of(connection, "pfp"))
    total = r[2] or 0
    if total == 0:
        connection.execute("DELETE FROM pfp_meta WHERE sensor=?", [sensor])
        connection.close()
        print(f"\nNothing ingested for {sensor}: every one of the {len(days)} "
              f"file(s) failed to read.", file=sys.stderr)
        for e in errors[:5]:
            print(f"  {e}", file=sys.stderr)
        return 1
    connection.execute("DELETE FROM pfp_meta WHERE sensor=?", [sensor])
    connection.execute("INSERT INTO pfp_meta VALUES (?,?,?,?,?,?,?,?,?)",
                [sensor, args.stat, NPOS, FRAME_MS, QMIN, QMAX, r[0], r[1], total])
    connection.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. {nrows:,} new frames, "
          f"{total:,} total. DB = {os.path.getsize(PFP_DB)/1e9:.2f} GB")
    if err:
        print(f"{err} file(s) failed to read; re-run to retry them.",
              file=sys.stderr)
    print("next: python serve.py     (optional: python ingest/compact_db.py "
          "first, to shrink the file)")
    # Non-zero for ANY unread file, matching psd_ingest. `err and nrows == 0`
    # reported success for a run that stored some days and dropped others, and
    # ingest_all.py branches on this to decide whether a step is done.
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
