"""
build_db.py: Tier 1 ingest and time-pyramid builder for the RF spectrum viewer.

Reads every Summaries CSV under ingest/csv (searched recursively, so the layout
fetch.py produced works as-is), loads the columns we care about into a single
DuckDB database, then precomputes coarse "zoom levels" so the viewer never has
to scan raw rows when you're looking at months or years at a time.

This database powers the zoomed-out summary layer. It is NOT required by
psd_ingest.py or pfp_ingest.py; those read their own CSVs directly.

Run once (re-run any time the CSVs change):

    py ingest/build_db.py
    py ingest/build_db.py --csv-dir /path/to/summaries
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cbrs_files                                    # noqa: E402
import chunk_io                                      # noqa: E402
import atlas                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root (scripts live in ingest/)
CSV_DIR = os.path.join(HERE, "csv")
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
DB_PATH = os.environ.get("SPECTRUM_DB") or os.path.join(DB_DIR, "spectrum.duckdb")

# The columns a Summaries CSV carries. This is MESSAGE TEXT, not the filter --
# find_csvs decides by filename through atlas.kind_of, so a PSD or PFP export
# never reaches the reader at all. It is quoted when a folder turns out to hold
# nothing usable, so the reader learns what was actually wanted.
NEEDED = ("sensor_name", "channel_frequency_mhz", "timestamp",
          "max", "median", "mean")

# Pyramid: bucket size in seconds -> table name. "raw" (native ~90s cadence) is
# kept as-is; these are the coarser pre-aggregated levels for zoomed-out views.
LEVELS = [
    ("lvl_m10", 600),      # 10 minutes
    ("lvl_h1", 3600),      # 1 hour
    ("lvl_h6", 21600),     # 6 hours
    ("lvl_d1", 86400),     # 1 day
]


def _has_rows(path):
    """Does this database actually hold summary rows? False if unreadable."""
    try:
        c = duckdb.connect(path, read_only=True)
    except Exception:                                      # noqa: BLE001
        return False
    try:
        if "raw" not in chunk_io.tables(c):
            return False
        return bool(c.execute("SELECT count(*) FROM raw").fetchone()[0])
    except Exception:                                      # noqa: BLE001
        return False
    finally:
        c.close()


def find_csvs(csv_dir):
    """Every CSV under csv_dir that is actually a Summaries export.

    The filter is the point, and it is a NAME filter first. This used to return
    every .csv underneath and hand each one to DuckDB's read_csv_auto to discover
    its columns, skipping the ones that turned out to be PSD or PFP exports. That
    is fine on a folder of summaries and catastrophic on a whole dataset root: on
    a network or on-demand filesystem -- Box Drive, OneDrive, a mounted share --
    opening a file is a DOWNLOAD, and the real CBRS tree holds ~44,600 PSD and
    PFP exports against 24 summaries files. Worse, they sort first, so every one
    of them was downloaded before the first usable file was reached. Measured:
    over an hour with zero bytes written and a 12 KB database, indistinguishable
    from a hang.

    atlas.kind_of decides PSD and PFP by filename and only reads a header for a
    CSV whose name matches neither -- so the ~44,600 cost nothing and just the
    two dozen candidates are opened. It is also the same classification
    ingest_all.py uses, so the two agree about what a summaries file is.
    """
    paths = cbrs_files.walk_ext(csv_dir, (".csv",))
    keep, other = [], []
    for p in paths:
        if atlas.kind_of(os.path.dirname(p), os.path.basename(p)) == "summaries":
            keep.append(p)
        else:
            other.append(p)
    if other:
        print(f"  {len(other):,} CSV file(s) under {os.path.abspath(csv_dir)} are "
              f"not Summaries exports (PSD and PFP exports have their own "
              f"scripts); {len(keep)} to read.", flush=True)
    return keep, other


def no_data(csv_dir):
    """Say what was searched for and how to get it, not just 'no CSV files'."""
    where = os.path.abspath(csv_dir)
    print(f"No CSV files found under {where}", file=sys.stderr)
    if not os.path.isdir(csv_dir):
        print("  That directory does not exist yet.", file=sys.stderr)
    print("\nThis script builds the zoomed-out summary layer from the CBRS\n"
          "Summaries CSVs. To download them into the right place:\n\n"
          "    python ingest/fetch.py <record-id> --filter Summaries \\\n"
          "        --dest ingest/csv --flat\n\n"
          "Or point at a copy you already have:\n\n"
          "    python ingest/build_db.py --csv-dir /path/to/summaries\n\n"
          "You do NOT need this database for the PSD or PFP layers; those read\n"
          "their own CSVs directly (python ingest/psd_ingest.py --list).",
          file=sys.stderr)
    return 1


def main():
    ap = argparse.ArgumentParser(
        description="Build spectrum.duckdb (the summary layer) from Summaries CSVs.")
    ap.add_argument("--csv-dir", default=CSV_DIR,
                    help=f"directory of Summaries CSVs (default {CSV_DIR})")
    args = ap.parse_args()

    files, other = find_csvs(args.csv_dir)
    if not files:
        if other:
            print(f"None of the {len(other):,} CSV file(s) under "
                  f"{os.path.abspath(args.csv_dir)} are Summaries exports.\n"
                  f"  A Summaries CSV has the columns: {', '.join(NEEDED)}.\n"
                  "  PSD and PFP exports belong to psd_ingest.py / pfp_ingest.py "
                  "instead.", file=sys.stderr)
            return 1
        return no_data(args.csv_dir)

    # Build to a NEW file and swap it in at the end, rather than deleting the
    # live one first. Deleting first means the working summary layer is gone from
    # the moment this starts: on the real dataset the rebuild reads ~9 GB of
    # monthly CSVs off a network drive, and for that whole window -- and forever,
    # if the run is interrupted -- the viewer has no zoomed-out layer and a 12 KB
    # stub where a 0.5 GB database was. Nothing about this step needs the old
    # file gone; it only needed somewhere to write. compact_db.py has always
    # worked this way.
    build_path = DB_PATH + ".build"
    for stale in (build_path, build_path + ".wal"):
        if os.path.exists(stale):
            os.remove(stale)
    # A WAL left behind by a previous run that crashed or was killed mid-build is
    # a separate file, and DuckDB replays it automatically on connect -- against
    # the brand-new, empty database this run just created. That replay re-issues
    # the old run's CREATE TABLE raw and collides with the one below ("Table with
    # name \"raw\" already exists!"), leaving the file unreadable. Starting from a
    # path with neither is what actually starts clean.
    connection = duckdb.connect(build_path)
    # Be polite with RAM on a laptop; DuckDB will spill to disk if needed.
    connection.execute("PRAGMA memory_limit='3GB'")
    connection.execute("PRAGMA threads=4")

    connection.execute("""
        CREATE TABLE raw (
            sensor VARCHAR,   -- sensor_name
            freq   DOUBLE,    -- channel_frequency_mhz
            t      DOUBLE,    -- epoch seconds (UTC)
            mx     DOUBLE,    -- max power (dBm)
            md     DOUBLE,    -- median power (dBm)
            mn     DOUBLE     -- mean power (dBm)
        )
    """)

    print(f"Ingesting {len(files)} CSV files...")
    t0 = time.time()
    used = skipped = 0
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        # read_csv_auto figures out the schema; we only pull the 6 columns we need.
        try:
            connection.execute("""
                INSERT INTO raw
                SELECT
                    sensor_name,
                    channel_frequency_mhz,
                    epoch(CAST(timestamp AS TIMESTAMPTZ)),
                    "max", median, mean
                FROM read_csv_auto(?, header=true,
                                   types={'channel_frequency_mhz':'DOUBLE',
                                          'max':'DOUBLE','median':'DOUBLE','mean':'DOUBLE'})
            """, [f])
        except duckdb.Error as e:
            # find_csvs already excluded PSD and PFP exports by filename, so a
            # failure here is a MALFORMED summaries file -- a truncated download,
            # a changed column name, a stray CSV whose header happened to mention
            # both sensor_name and channel_frequency_mhz. Say which file and keep
            # going: one bad month should not cost the other twenty-three.
            skipped += 1
            first = str(e).strip().splitlines()[0]
            print(f"  [{i:2}/{len(files)}] {name:18}  skipped: {first}")
            continue
        used += 1
        n = connection.execute("SELECT count(*) FROM raw").fetchone()[0]
        print(f"  [{i:2}/{len(files)}] {name:18}  total rows: {n:,}  ({time.time()-t0:.0f}s)")

    if used == 0:
        connection.close()
        os.remove(build_path)
        if os.path.exists(build_path + ".wal"):
            os.remove(build_path + ".wal")
        print(f"\nNone of the {len(files)} CSV file(s) under "
              f"{os.path.abspath(args.csv_dir)} are Summaries exports.\n"
              f"  A Summaries CSV has the columns: {', '.join(NEEDED)}.\n"
              "  PSD and PFP exports belong to psd_ingest.py / pfp_ingest.py "
              "instead.", file=sys.stderr)
        return 1
    if skipped:
        print(f"  ({skipped} file(s) skipped, not Summaries exports)")

    print("Building pyramid levels...")
    for tbl, bucket in LEVELS:
        connection.execute(f"""
            CREATE TABLE {tbl} AS
            SELECT sensor, freq,
                   floor(t/{bucket})*{bucket} AS t,
                   max(mx) AS mx,      -- true peak of peaks
                   avg(md) AS md,      -- approx (mean of medians), fine for overview
                   avg(mn) AS mn,
                   count(*) AS c
            FROM raw
            GROUP BY sensor, freq, floor(t/{bucket})*{bucket}
            ORDER BY sensor, t
        """)
        rows = connection.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:10} bucket={bucket:>6}s  rows: {rows:,}")

    # Keep raw sorted for fast range scans at full zoom.
    connection.execute("CREATE TABLE raw_sorted AS SELECT * FROM raw ORDER BY sensor, t")
    connection.execute("DROP TABLE raw")
    connection.execute("ALTER TABLE raw_sorted RENAME TO raw")

    # Metadata the viewer needs on load.
    connection.execute("""
        CREATE TABLE meta AS
        SELECT sensor,
               min(t) AS t_min, max(t) AS t_max,
               count(*) AS n
        FROM raw GROUP BY sensor ORDER BY sensor
    """)

    print("\nSensors:")
    for sensor, tmin, tmax, n in connection.execute(
            "SELECT sensor, t_min, t_max, n FROM meta").fetchall():
        days = (tmax - tmin) / 86400
        print(f"  {sensor:22} {n:>12,} rows  spanning {days:6.1f} days")

    freqs = [r[0] for r in connection.execute(
        "SELECT DISTINCT freq FROM raw ORDER BY freq").fetchall()]
    print(f"\nFrequencies (MHz): {freqs}")

    connection.execute("CHECKPOINT")     # so nothing is left only in the WAL
    connection.close()
    if os.path.exists(build_path + ".wal"):
        # Same hazard compact_db.py guards: renaming the .duckdb while its WAL
        # sits beside it either strands the data or makes the file unopenable.
        print(f"\nNOT replacing {os.path.basename(DB_PATH)}: "
              f"{os.path.basename(build_path)}.wal still exists, so part of the "
              f"build is not in the file yet. The previous database is untouched.",
              file=sys.stderr)
        return 1
    if os.path.exists(DB_PATH):
        # Keep the outgoing file only if it HOLDS something. A failed earlier run
        # can leave a 12 KB stub here, and backing that up just litters the folder
        # with a file whose name promises a database and delivers nothing -- while
        # a real 2 GB summary layer is exactly what you want a backup of.
        if _has_rows(DB_PATH):
            bak = DB_PATH + ".bak"
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(DB_PATH, bak)
            if os.path.exists(DB_PATH + ".wal"):
                os.replace(DB_PATH + ".wal", bak + ".wal")
            print(f"  previous {os.path.basename(DB_PATH)} "
                  f"({os.path.getsize(bak)/1e9:.2f} GB) kept as "
                  f"{os.path.basename(bak)} -- delete it when you are happy")
        else:
            os.remove(DB_PATH)
            if os.path.exists(DB_PATH + ".wal"):
                os.remove(DB_PATH + ".wal")
            print(f"  replaced an empty {os.path.basename(DB_PATH)} "
                  f"(no backup kept: it held no rows)")
    os.replace(build_path, DB_PATH)
    size_gb = os.path.getsize(DB_PATH) / 1e9
    print(f"\nDone in {time.time()-t0:.0f}s. Database: {DB_PATH} ({size_gb:.2f} GB)")
    print("next: python serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
