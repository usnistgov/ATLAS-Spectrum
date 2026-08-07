"""
psd_ingest.py: ingest PSD numbers into a compact, fast store.

Each capture's 2250-bin spectrum is quantized to uint8 (0.35 dB steps over
[-180,-90] dBm/Hz) and stored as one BLOB row: (sensor, t, spec). That's ~1
row per capture (~180k) instead of 398M long-form rows, so ingest skips the
expensive UNPIVOT, the DB is ~4x smaller, and rendering is just a fetch + numpy
bin. Re-render at any frequency zoom stays crisp (the numbers are preserved).

Source CSVs are found by walking --root (default $SEA_DATA_ROOT, else
./SEA-DATA) and matching <YYYY-MM-DD>_<sensor>_<stat>.csv anywhere underneath,
so it does not matter whether they sit in a PSD/ folder, in the layout
fetch.py produced, or loose in one directory.

The server reads this file directly. compact_db.py repacks it into a smaller
chunked form and swaps that in; that step is now optional and only affects
size and speed, not whether the viewer works. Re-running against a database
that is already compacted appends in the chunked shape rather than the row
shape, so next month's data lands where the server will read it.

    py psd_ingest.py --list                        # what is on disk
    py psd_ingest.py CBBT-Directional              # one sensor
    py psd_ingest.py CBBT-Directional --limit 5    # quick test
    py psd_ingest.py --root /path/to/SEA-DATA      # explicit source directory
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cbrs_files                                    # noqa: E402
import chunk_io                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root (scripts live in ingest/)
# Where the CBRS source CSVs live. Set SEA_DATA_ROOT to your copy, or pass
# --root; defaults to ./SEA-DATA in the repo so a fresh clone never points at
# someone else's disk.
DATA_ROOT = os.environ.get("SEA_DATA_ROOT", os.path.join(ROOT, "SEA-DATA"))
# abspath, as serve.py already does: a RELATIVE ATLAS_DB_DIR resolves against
# whatever directory each step happens to be run from, so the ingest wrote
# ./mydb/psd.duckdb and compact_db.py looked for it under a different cwd,
# reported "not found -- build it first", and exited 0. Same variable, two files.
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
PSD_DB = os.environ.get("PSD_DB") or os.path.join(DB_DIR, "psd.duckdb")

F0 = 3530040000.0    # first PSD bin (Hz)
DF = 80000.0         # bin spacing (Hz)
NF = 2250            # bins
QMIN, QMAX = -180.0, -90.0   # uint8 quantization range (dBm/Hz)

# <YYYY-MM-DD>_<sensor>_<stat>.csv . The sensor match is lazy and the stat has
# no underscore, so sensor names containing underscores still resolve.
NAME_RE = re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})_(?P<sensor>.+?)_"
                     r"(?P<stat>[A-Za-z0-9]+)\.csv$")
PATTERN = "<YYYY-MM-DD>_<sensor>_<stat>.csv"

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def discover(root, stat=None, collisions=None):
    return cbrs_files.discover(root, NAME_RE, stat, collisions)


def stats_present(root):
    return cbrs_files.stats_present(root, NAME_RE)


def read_specs(path, csv_connection):
    """CSV -> (epoch_seconds[n], quantized uint8 [n,2250])."""
    ts, _keys, Q = cbrs_files.read_quantized(
        path, csv_connection, NF, QMIN, QMAX, "dBm/Hz", "spectrum")
    return ts, Q


def open_db(path):
    return cbrs_files.open_db(path, duckdb)


def _rollback(connection, app):
    return chunk_io.rollback_day(connection, app)


def main():
    ap = argparse.ArgumentParser(
        description="Ingest CBRS PSD CSVs into psd.duckdb.")
    ap.add_argument("sensor", nargs="?", default=None,
                    help="sensor name; omit when only one is on disk")
    ap.add_argument("--root", default=DATA_ROOT,
                    help=f"directory holding the source CSVs (default {DATA_ROOT})")
    ap.add_argument("--stat", default="max",
                    help="which statistic to ingest (default max)")
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
                                    PATTERN, "PSD"))

    if args.list:
        print(f"{os.path.abspath(root)} (stat: {args.stat})\n")
        for s in sorted(found):
            days = sorted(found[s])
            print(f"  {s:24} {len(days):>5} day(s)  {days[0]} .. {days[-1]}")
        print(f"\nstats available: {', '.join(stats_present(root))}")
        print("\nnext: python ingest/psd_ingest.py <sensor>")
        return 0

    # Each statistic gets its own database, next to the max one: psd.duckdb for
    # max (the path that always existed, unchanged), psd_median.duckdb and
    # psd_mean.duckdb for the others. Separate files rather than a schema
    # change, so every existing database keeps working untouched and serve.py
    # simply offers whichever statistics have been ingested.
    global PSD_DB
    if args.stat != "max":
        base_dir = os.path.dirname(PSD_DB) or "."
        PSD_DB = os.path.join(base_dir, f"psd_{args.stat}.duckdb")
        print(f"stat '{args.stat}' -> {os.path.basename(PSD_DB)}")

    sensor, why = cbrs_files.resolve_sensor(args.sensor, found, root)
    if why:
        sys.exit(why)
    if args.sensor is None:
        print(f"one sensor on disk; ingesting {sensor}")

    by_day = found[sensor]
    days = sorted(by_day)
    print(f"{sensor}: {len(days)} day(s) on disk (compact uint8 BLOB)")

    connection = open_db(PSD_DB)
    connection.execute("CREATE TABLE IF NOT EXISTS psd (sensor VARCHAR, t DOUBLE, spec BLOB)")
    connection.execute("""CREATE TABLE IF NOT EXISTS psd_meta (
        sensor VARCHAR, f0 DOUBLE, df DOUBLE, nf INT, qmin DOUBLE, qmax DOUBLE,
        t_min DOUBLE, t_max DOUBLE, captures BIGINT)""")
    # This database may already be compacted: the datasets grow, so running the
    # ingest again next month usually means appending to a file compact_db.py
    # has rewritten into chunks. Write in whichever shape is already there --
    # writing rows into a compacted file leaves both schemas in it, and
    # serve.py, preferring the chunk table, would never draw the new month.
    kind = chunk_io.schema_of(connection, "psd")
    if kind == "chunk":
        print("  this database is compacted; appending new days as chunks")
    # Resumable: which days are already stored. The rule is subtler than it
    # looks -- UTC bucketing on both sides, and each file tested against its OWN
    # first capture rather than the day its name carries, because a CBRS export
    # runs past the next midnight. chunk_io.day_is_ingested and .already_have
    # carry the full argument and what each shortcut costs; do not reason about
    # this from here.
    times = chunk_io.stored_times(connection, "psd", sensor, kind)
    stored_ms = {int(round(t * 1000)) for t in times}
    per_day = chunk_io.captures_per_day(times)
    todo = [d for d in days
            if not chunk_io.day_is_ingested(d, per_day, stored_ms,
                                            lambda: cbrs_files.first_capture_time(by_day[d]))]
    # Count the days THIS RUN is skipping, not every day the sensor has stored:
    # the database spans days that are not under this --root at all, and counting
    # those is how "1 day on disk / 2 day(s) already ingested" -- more skipped
    # than found -- came about.
    if len(todo) < len(days):
        print(f"  resuming: {len(days) - len(todo)} of {len(days)} day(s) "
              f"already ingested, skipping those")
    # --limit caps the days this run reads, applied after the resume filter.
    # Capping `days` first re-selects the same already-ingested days on every
    # run, so a limited resume can never advance past the first N days.
    if args.limit:
        todo = todo[:args.limit]
    csv_connection = duckdb.connect()   # separate handle for CSV reads

    t0 = time.time()
    done = err = caps = 0
    errors = []
    app = (chunk_io.ChunkAppender(connection, "psd", sensor, "specs")
           if kind == "chunk" else None)
    # A day is written inside ONE transaction, and that is load-bearing rather
    # than tidiness. DuckDB autocommits per parameter set, so an interrupted
    # executemany left an arbitrary PREFIX of the day committed -- and the resume
    # check then sees hundreds of captures on that UTC day and calls it done, so
    # the rest is skipped forever. Measured: a SIGKILL, a Ctrl+C, or a full disk
    # partway through turned 2,349 captures into 1,637 and the NEXT run reported
    # "3 of 3 day(s) already ingested", exit code 0. Both resume tests ask "was
    # this file started", never "was it finished"; wrapping the day makes those
    # the same question, because a day that did not finish contributes nothing.
    interrupts = tuple(x for x in (KeyboardInterrupt,
                                   getattr(duckdb, "InterruptException", None))
                       if x is not None)
    for i, day in enumerate(todo, 1):
        path = by_day[day]
        try:
            connection.execute("BEGIN TRANSACTION")
            ts, Q = read_specs(path, csv_connection)
            if not len(ts):
                # A header-only export used to count as a day done (err=0,
                # exit 0) whenever the database was already compacted, so the
                # real file for that day could never be picked up again.
                raise ValueError("the file has a header but no data rows")
            if app is not None:
                app.add(ts, [Q[j] for j in range(len(ts))])
                app.flush()          # inside this day's transaction
            else:
                connection.executemany(
                    "INSERT INTO psd VALUES (?,?,?)",
                    [(sensor, float(ts[j]), Q[j].tobytes()) for j in range(len(ts))])
            connection.execute("COMMIT")
            done += 1; caps += len(ts)
        except interrupts:
            # DuckDB turns Ctrl+C into InterruptException, which is an Exception,
            # so the handler below used to swallow it and carry on to the next
            # day -- Ctrl+C did not stop the run, and landing inside a chunk
            # flush instead committed a partial day.
            _rollback(connection, app)
            print(f"\ninterrupted during {os.path.basename(path)}; that day was "
                  f"rolled back. Re-run to continue where this left off.",
                  file=sys.stderr)
            connection.close()
            return 130
        except Exception as e:
            _rollback(connection, app)
            err += 1
            errors.append(f"{os.path.basename(path)}: {e}")
            print(f"  ERR {os.path.basename(path)}: {e}")
        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(todo)}] caps={caps:,} done={done} err={err} "
                  f"{rate:.1f} days/s")

    rng = chunk_io.stored_span(connection, "psd", sensor,
                               chunk_io.schema_of(connection, "psd"))
    total = rng[2] or 0
    if total == 0:
        # Never leave a meta row claiming a sensor that has no spectra: the
        # server would advertise the layer and then have nothing to draw.
        connection.execute("DELETE FROM psd_meta WHERE sensor=?", [sensor])
        connection.close()
        print(f"\nNothing ingested for {sensor}: every one of the {len(todo)} "
              f"file(s) failed to read.", file=sys.stderr)
        for e in errors[:5]:
            print(f"  {e}", file=sys.stderr)
        return 1
    connection.execute("DELETE FROM psd_meta WHERE sensor=?", [sensor])
    connection.execute("INSERT INTO psd_meta VALUES (?,?,?,?,?,?,?,?,?)",
                [sensor, F0, DF, NF, QMIN, QMAX, rng[0], rng[1], total])
    connection.close()
    size = os.path.getsize(PSD_DB) / 1e9
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. {sensor}: {caps:,} new "
          f"captures, {total:,} total. DB {PSD_DB} = {size:.2f} GB (all sensors)")
    if err:
        print(f"{err} file(s) failed to read; re-run to retry them.",
              file=sys.stderr)
    print("next: python serve.py     (optional: python ingest/compact_db.py "
          "first, to shrink the file)")
    # Any unread file means this sensor is missing days, so exit non-zero even
    # when other files succeeded. `err and caps == 0` reported success for a
    # run that dropped half its input, and atlas.py and make_sample.py branch
    # on this to decide whether the step is done.
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
