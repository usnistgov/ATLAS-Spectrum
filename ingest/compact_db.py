"""compact_db.py: repack the live DBs into much smaller files (lossless).

  spectrum.duckdb : dBm DOUBLE -> SMALLINT (dBm*10), t -> BIGINT    ~1.9 -> ~0.7 GB
  psd.duckdb      : per-capture rows -> zlib chunks of 256 spectra  ~5.4 -> ~3.3 GB
  pfp.duckdb      : per-capture rows -> zlib chunks of 1024 frames  ~14.3 -> ~5.5 GB
                    (consecutive same-channel frames compress 3.1x; measured)

Writes *_c.duckdb build files next to the originals, then moves each one into
place, keeping the original as <name>.duckdb.bak. Pass --no-swap to skip that
last step. Resumable: finished units are recorded in a `done` table, a crashed
partial unit is deleted and redone.

This step is optional. serve.py reads the uncompacted files the ingest scripts
write, so compaction only makes them smaller and faster.

    py compact_db.py
    py compact_db.py --no-swap
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
ROOT = os.path.dirname(HERE)   # repo root (scripts live in ingest/)
# Where the .duckdb files live; override to compact a copy elsewhere.
# abspath, matching serve.py: a RELATIVE ATLAS_DB_DIR resolves against each
# step's cwd, so the ingest wrote ./mydb/psd.duckdb and this script looked in a
# different ./mydb, said "not found -- build it first", and exited 0.
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)
Z = 9                # zlib level -- decompression cost is the same at every
                     # level, so max level is a free size win. Measured on live
                     # PFP chunks: 2.6% smaller than z6, identical read speed.
PSD_CHUNK = 256      # spectra per chunk  (256*2250 = 576 KB raw)
PFP_CHUNK = 1024     # frames per chunk   (1024*560 = 573 KB raw)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _finish(dst):
    """Flush a build file to its own .duckdb and close it.

    DuckDB writes into a WAL and checkpoints on close -- but a close that CANNOT
    checkpoint (no space left) does not raise, so the function above reported the
    build "complete" while its rows were still only in <name>_c.duckdb.wal. The
    row-count guard was then fooled too, because opening the build file
    read-only replays that same WAL and counts them. swap_in renamed the .duckdb
    and orphaned the .wal, and the live database ended up with ZERO tables --
    exit code 0, "DONE ALL". An explicit CHECKPOINT turns that into an exception
    here, where it can be reported and nothing has been swapped yet.
    """
    dst.execute("CHECKPOINT")
    dst.close()


def _q(path):
    """A path safe to interpolate into DuckDB SQL.

    ATTACH takes the file as a string literal, and the path was interpolated
    raw -- so a single quote anywhere in it (`~/Sean's data`, `/Users/o'brien`)
    aborted compaction with a ParserException after the ingest had already
    written correct data, leaving a permanently uncompacted database and an
    unactionable traceback.
    """
    return path.replace("'", "''")


def _prep(dst, ddl):
    dst.execute("CREATE TABLE IF NOT EXISTS done (k VARCHAR)")
    dst.execute(ddl)


def _skip(dst, key):
    return dst.execute("SELECT 1 FROM done WHERE k=?", [key]).fetchone() is not None


def _missing(src_p):
    """True (with a clear message) when the source DB hasn't been built yet.
    Keeps a fresh clone from dying on a raw DuckDB traceback."""
    if os.path.exists(src_p):
        return False
    log(f"  skip: {os.path.basename(src_p)} not found in {DB_DIR}. "
        "Build it with the matching ingest script first (see README).")
    return True


def compact_spectrum():
    """-> True when spectrum_c.duckdb is complete and safe to swap in."""
    src_p = os.path.join(DB_DIR, "spectrum.duckdb")
    if _missing(src_p):
        return False
    # This rewrite multiplies every dBm by 10 into a SMALLINT, and it had NO
    # guard against running twice. Measured over three consecutive runs:
    # -85.9 dBm -> -859 (correct) -> -8590, which serve.py then reads back as
    # -859.0 dBm because it takes the scale off the column type and the type is
    # still SMALLINT -- so nothing anywhere notices -- and then an
    # OutOfRangeException that aborts the run before psd and pfp are reached.
    # Two runs was enough to lose the original: swap_in overwrites the .bak.
    if _spectrum_is_scaled(src_p):
        log("  skip: spectrum.duckdb is already compacted (dBm*10 SMALLINT)")
        return False
    _drop_stale_build("spectrum", scaled=True)
    dst = duckdb.connect(os.path.join(DB_DIR, "spectrum_c.duckdb"))
    dst.execute("CREATE TABLE IF NOT EXISTS done (k VARCHAR)")
    dst.execute(f"ATTACH '{_q(src_p)}' AS s (READ_ONLY)")
    dst.execute("DROP TABLE IF EXISTS meta")
    dst.execute("CREATE TABLE meta AS SELECT * FROM s.meta")
    for t in ("lvl_d1", "lvl_h6", "lvl_h1", "lvl_m10", "raw"):
        if _skip(dst, t):
            continue
        t0 = time.time()
        dst.execute(f"DROP TABLE IF EXISTS {t}")
        # dBm*10 as SMALLINT: the API already rounds to 0.1 dBm, so lossless
        # as served. ORDER BY improves zonemap pruning + bitpacking.
        dst.execute(f"""CREATE TABLE {t} AS
            SELECT sensor, freq, CAST(t AS BIGINT) AS t,
                   CAST(ROUND(mx*10) AS SMALLINT) AS mx,
                   CAST(ROUND(md*10) AS SMALLINT) AS md,
                   CAST(ROUND(mn*10) AS SMALLINT) AS mn
            FROM s.{t} ORDER BY sensor, t""")
        dst.execute("INSERT INTO done VALUES (?)", [t])
        log(f"  spectrum.{t} in {time.time()-t0:.0f}s")
    _finish(dst)
    log("spectrum_c.duckdb complete")
    return True


def _spectrum_is_scaled(path):
    """True when the summary table already stores dBm*10 as an integer.

    The same test serve.py makes (serve._dbm_div): the scale is a property of
    the column type, so it can be read off the file instead of assumed.
    """
    try:
        c = duckdb.connect(path, read_only=True)
    except Exception:                                      # noqa: BLE001
        return False
    try:
        for tbl in ("lvl_h1", "lvl_m10", "lvl_d1", "raw"):
            r = c.execute("SELECT data_type FROM duckdb_columns() WHERE "
                          "table_name=? AND column_name='mx'", [tbl]).fetchone()
            if r:
                return "INT" in r[0].upper()
        return False
    except Exception:                                      # noqa: BLE001
        return False
    finally:
        c.close()


def _src_stamp(path, rows_table):
    """A fingerprint of the source database this build file was started from.

    Just enough to notice the source has changed: how many rows the source table
    held and its newest timestamp. Cheap, and both move whenever a month is
    added.
    """
    try:
        c = duckdb.connect(path, read_only=True)
    except Exception:                                      # noqa: BLE001
        return None
    try:
        if rows_table not in _tables(c):
            return None
        r = c.execute(f"SELECT count(*), max(t) FROM {rows_table}").fetchone()
        return f"src:{rows_table}:{r[0]}:{r[1]}"
    except Exception:                                      # noqa: BLE001
        return None
    finally:
        c.close()


def _stamp_build(dst, stamp):
    if stamp and not _skip(dst, stamp):
        dst.execute("DELETE FROM done WHERE k LIKE 'src:%'")
        dst.execute("INSERT INTO done VALUES (?)", [stamp])


def _stamp_mismatch(built, stamp):
    """True when this build file was started from a DIFFERENT source than now.

    `_live_moved_on` only recognises "the live database is already chunked". It
    cannot see the case that actually happens: compaction is interrupted, the
    normal monthly ingest then ADDS a day to the still-row-shaped live database,
    and the old build file's `done` table makes the next compaction skip every
    sensor it already has, call itself complete, and swap the new days away.
    Measured: 21,904 captures down to 20,348, exit code 0.
    """
    if stamp is None:
        return False
    try:
        c = duckdb.connect(built, read_only=True)
    except Exception:                                      # noqa: BLE001
        return False
    try:
        if "done" not in _tables(c):
            return False
        rows = [r[0] for r in c.execute(
            "SELECT k FROM done WHERE k LIKE 'src:%'").fetchall()]
        if not rows:
            return True        # from before stamping; cannot be trusted to match
        return rows[0] != stamp
    except Exception:                                      # noqa: BLE001
        return False
    finally:
        c.close()


def _drop_stale_build(name, scaled=False, rows_table=None):
    """Remove a leftover <name>_c.duckdb that the live database has moved past.

    The build file carries a `done` table so an interrupted compaction can
    resume. That is only meaningful while the live database is still the shape
    the build file was started from. Once the live database has been compacted
    and then ADDED TO -- which is the normal monthly cycle -- an old build file
    is a time machine: `done` makes compaction skip every sensor it already has,
    declare itself complete, and swap months of newer data away. Measured: 800
    captures back to 300, and a sensor left with chunks but no meta row.

    A build file is left alone only when it could still be a legitimate resume.
    """
    built = _built_path(name)
    if not os.path.exists(built):
        return
    live = _live_path(name)
    if not os.path.exists(live):
        return
    if scaled:
        stale = _spectrum_is_scaled(live)
    else:
        stale = (_live_moved_on(live, rows_table)
                 or _stamp_mismatch(built, _src_stamp(live, rows_table)))
    if stale:
        try:
            os.remove(built)
            log(f"  removed a stale {name}_c.duckdb (the live database has moved "
                f"past it; a resume would have rolled data back)")
        except OSError as e:
            log(f"  WARNING: {name}_c.duckdb looks stale but could not be "
                f"removed: {e}. Delete it before compacting.")


def _live_moved_on(live, rows_table):
    """True when the live database is already in the chunk shape."""
    try:
        c = duckdb.connect(live, read_only=True)
    except Exception:                                      # noqa: BLE001
        return False
    try:
        tables = _tables(c)
        if rows_table not in tables:
            return True
        return c.execute(f"SELECT count(*) FROM {rows_table}").fetchone()[0] == 0
    except Exception:                                      # noqa: BLE001
        return False
    finally:
        c.close()


def _mixed_shape(connection, rows_table, chunk_table, label):
    """True when BOTH shapes hold data -- refuse rather than guess.

    Compaction reads sensors from the ROW table only and copies nothing from the
    chunk table, so compacting this state writes a file containing only the
    row-derived sensors and drops every chunked one. Measured: 1280 captures and
    their meta row destroyed. The emptiness backstop cannot see it, because the
    output is not empty. Nothing in the current pipeline creates this shape, but
    a hand-merged or legacy database looks exactly like it.
    """
    tables = _tables(connection)
    if rows_table not in tables or chunk_table not in tables:
        return False
    rows = connection.execute(f"SELECT count(*) FROM {rows_table}").fetchone()[0]
    chunks = connection.execute(f"SELECT count(*) FROM {chunk_table}").fetchone()[0]
    if rows and chunks:
        log(f"  REFUSING to compact {label}: it holds data in BOTH the "
            f"{rows_table} ({rows:,} rows) and {chunk_table} ({chunks:,} chunks) "
            f"shapes. Compaction reads only {rows_table} and would drop the "
            f"chunked data. Ingest into a fresh database instead.")
        return True
    return False


def _already_compact(connection, rows_table, label):
    """True when there is nothing to compact -- so don't.

    Two ways a database is already done, and the second one caused real data
    loss before it was handled:

    1. The row table is gone. `SELECT DISTINCT sensor FROM <rows>` then raised a
       bare CatalogException and took the whole run down with it, including the
       databases it had not reached yet.

    2. The row table EXISTS BUT IS EMPTY. Ingesting into an already-compacted
       database leaves exactly that: the ingest appends chunks and the row table
       stays behind with zero rows. Compaction then read no sensors, wrote a
       perfectly valid _c file containing nothing, and swapped it over the real
       chunks -- silently emptying the database. Measured: 72 captures to 0.

    The rule that covers both is simply "no rows to compact means no
    compaction". Compacting nothing can never do anything but harm.
    """
    tables = _tables(connection)
    if rows_table not in tables:
        log(f"  skip: {label} is already compacted")
        return True
    n = connection.execute(f"SELECT count(*) FROM {rows_table}").fetchone()[0]
    if n == 0:
        log(f"  skip: {label} has an empty {rows_table} table -- already compacted")
        return True
    return False


_tables = chunk_io.tables      # one definition, in the module that owns the schema


def _copy_meta(dst, src_path, table, rows_table=None):
    """Copy <table> across, dropping rows for sensors with no data behind them.

    The metadata and the data it describes can fall out of step: a database
    rebuilt or swapped underneath its own metadata keeps the old meta rows,
    and this function used to copy them forward verbatim. The result is a
    meta row advertising captures that are not in the file -- with a valid
    t_min/t_max, so nothing downstream notices. serve.py then reports the
    layer as present, the viewer enables it, and every window comes back
    empty: the layer silently never draws.

    Compaction is the natural place to catch this, because it is already
    reading both tables. Sensors are taken from the data table, so an orphan
    meta row is dropped rather than blessed and carried into the compacted
    file.
    """
    dst.execute(f"ATTACH '{_q(src_path)}' AS msrc (READ_ONLY)")
    dst.execute(f"DROP TABLE IF EXISTS {table}")
    dst.execute(f"CREATE TABLE {table} AS SELECT * FROM msrc.{table}")
    if rows_table:
        orphans = [r[0] for r in dst.execute(
            f"SELECT m.sensor FROM {table} m WHERE NOT EXISTS "
            f"(SELECT 1 FROM msrc.{rows_table} d WHERE d.sensor = m.sensor)"
        ).fetchall()]
        if orphans:
            dst.execute(f"DELETE FROM {table} WHERE sensor NOT IN "
                        f"(SELECT DISTINCT sensor FROM msrc.{rows_table})")
            log(f"  dropped {len(orphans)} {table} row(s) with no {rows_table} "
                f"data behind them: {', '.join(sorted(orphans))}")
            log("  re-run the matching ingest for those sensors if that is "
                "not what you expected.")
    dst.execute("DETACH msrc")
    # Deliberately NOT recorded in `done`. Marking it made a resumed compaction
    # skip the copy, so a build file started last month carried last month's
    # psd_meta forward over this month's data -- stale t_max, stale capture
    # counts, for every sensor at once. Copying a metadata table is cheap; the
    # only thing the `done` marker bought was that staleness.


def _live_path(name):
    """The live database for one layer, honouring PSD_DB for the psd family."""
    if name.startswith("psd"):
        return _psd_path(name)
    return os.path.join(DB_DIR, f"{name}.duckdb")


def _built_path(name):
    """The build file, always beside the live database it was made from."""
    return os.path.join(os.path.dirname(_live_path(name)) or DB_DIR,
                        f"{name}_c.duckdb")


def _psd_path(name):
    """Where this statistic's database lives.

    psd_ingest.py and build_psd_levels.py both honour PSD_DB; this script only
    looked at ATLAS_DB_DIR, so with PSD_DB set the ingest wrote 60 captures to
    the custom path, the levels were built there, and compaction printed
    "psd.duckdb not found ... build it first" and exited 0 -- advising the user
    to do the thing they had just done.
    """
    override = os.environ.get("PSD_DB")
    if override and name == "psd":
        return os.path.abspath(override)
    if override:
        base = os.path.dirname(os.path.abspath(override)) or DB_DIR
        return os.path.join(base, f"{name}.duckdb")
    return os.path.join(DB_DIR, f"{name}.duckdb")


def compact_psd(name="psd"):
    """-> True only if every sensor round-tripped with the same row count.

    `name` is the database's base name: `psd` for the max statistic,
    `psd_median` / `psd_mean` for the sibling databases psd_ingest.py --stat
    writes. Same schema inside, so one implementation compacts them all.
    """
    src_p = _psd_path(name)
    if _missing(src_p):
        return False
    src = duckdb.connect(src_p, read_only=True)
    if _mixed_shape(src, "psd", "psd_chunk", f"{name}.duckdb"):
        src.close()
        return False
    if _already_compact(src, "psd", f"{name}.duckdb"):
        src.close()
        return False        # nothing was verified, so nothing may be swapped
    src.close()
    _drop_stale_build(name, rows_table="psd")
    src = duckdb.connect(src_p, read_only=True)
    dst = duckdb.connect(_built_path(name))
    _prep(dst, """CREATE TABLE IF NOT EXISTS psd_chunk (
        sensor VARCHAR, t0 DOUBLE, t1 DOUBLE, n INT, times BLOB, specs BLOB)""")
    _stamp_build(dst, _src_stamp(src_p, "psd"))
    _copy_meta(dst, src_p, "psd_meta", "psd")
    sensors = [r[0] for r in src.execute("SELECT DISTINCT sensor FROM psd ORDER BY 1").fetchall()]
    bad = []
    for s in sensors:
        if _skip(dst, s):
            log(f"  psd {s}: already done")
            continue
        t0 = time.time()
        dst.execute("DELETE FROM psd_chunk WHERE sensor=?", [s])   # partial from a crash
        cur = src.execute("SELECT t, spec FROM psd WHERE sensor=? ORDER BY t", [s])
        total = 0
        dst.execute("BEGIN")
        while True:
            rows = cur.fetchmany(PSD_CHUNK)
            if not rows:
                break
            ts = np.array([r[0] for r in rows], dtype=np.float64)
            dst.execute("INSERT INTO psd_chunk VALUES (?,?,?,?,?,?)",
                        [s, float(ts[0]), float(ts[-1]), len(rows),
                         zlib.compress(ts.tobytes(), Z),
                         zlib.compress(b"".join(r[1] for r in rows), Z)])
            total += len(rows)
        dst.execute("COMMIT")
        want = src.execute("SELECT count(*) FROM psd WHERE sensor=?", [s]).fetchone()[0]
        if total != want:
            log(f"  psd {s}: MISMATCH {total} != {want}. NOT marking done")
            bad.append(s)
            continue
        dst.execute("INSERT INTO done VALUES (?)", [s])
        log(f"  psd {s}: {total} spectra in {time.time()-t0:.0f}s")
    # Carry the coarse pyramid across. psd_lvl is bucketed spectra keyed by
    # sensor and time -- it describes the same measurements whichever on-disk
    # shape holds them, so compaction has no reason to invalidate it. It used to
    # be dropped silently: build the levels and then compact (the order
    # an earlier one-command script used) and every level row was discarded, leaving wide
    # windows to the slow capture path with nothing on screen to say why. Copying
    # it makes the two steps order-independent, which is the real fix -- a
    # pipeline should not have a correct order you can only learn by measuring.
    for t in ("psd_lvl", "psd_lvl_done"):
        if t in _tables(src):
            dst.execute(f"DROP TABLE IF EXISTS {t}")
            dst.execute(f"ATTACH '{_q(src_p)}' AS lsrc (READ_ONLY)")
            dst.execute(f"CREATE TABLE {t} AS SELECT * FROM lsrc.{t}")
            dst.execute("DETACH lsrc")
            n = dst.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            log(f"  {name} {t}: carried {n:,} row(s) across")
    src.close(); _finish(dst)
    if bad:
        log(f"{name}_c.duckdb INCOMPLETE for {', '.join(bad)}; not swapping it in")
        return False
    log(f"{name}_c.duckdb complete")
    return True


def compact_pfp():
    """-> True only if every sensor round-tripped with the same row count."""
    src_p = os.path.join(DB_DIR, "pfp.duckdb")
    if _missing(src_p):
        return False
    src = duckdb.connect(src_p, read_only=True)
    if _mixed_shape(src, "pfp", "pfp_chunk", "pfp.duckdb"):
        src.close()
        return False
    if _already_compact(src, "pfp", "pfp.duckdb"):
        src.close()
        return False        # nothing was verified, so nothing may be swapped
    src.close()
    _drop_stale_build("pfp", rows_table="pfp")
    src = duckdb.connect(src_p, read_only=True)
    dst = duckdb.connect(os.path.join(DB_DIR, "pfp_c.duckdb"))
    _prep(dst, """CREATE TABLE IF NOT EXISTS pfp_chunk (
        sensor VARCHAR, freq DOUBLE, t0 DOUBLE, t1 DOUBLE, n INT, times BLOB, frames BLOB)""")
    _stamp_build(dst, _src_stamp(src_p, "pfp"))
    _copy_meta(dst, src_p, "pfp_meta", "pfp")
    sensors = [r[0] for r in src.execute("SELECT DISTINCT sensor FROM pfp ORDER BY 1").fetchall()]
    bad = []
    for s in sensors:
        if _skip(dst, s):
            log(f"  pfp {s}: already done")
            continue
        t0 = time.time()
        dst.execute("DELETE FROM pfp_chunk WHERE sensor=?", [s])
        cur = src.execute("SELECT freq, t, frame FROM pfp WHERE sensor=? ORDER BY freq, t", [s])
        cf, ts, fr, total = None, [], [], 0

        def flush():
            if not ts:
                return
            ta = np.array(ts, dtype=np.float64)
            dst.execute("INSERT INTO pfp_chunk VALUES (?,?,?,?,?,?,?)",
                        [s, cf, float(ta[0]), float(ta[-1]), len(ts),
                         zlib.compress(ta.tobytes(), Z),
                         zlib.compress(b"".join(fr), Z)])

        dst.execute("BEGIN")
        while True:
            rows = cur.fetchmany(16384)
            if not rows:
                break
            for f, t, frame in rows:
                if f != cf or len(ts) >= PFP_CHUNK:
                    flush()
                    cf, ts, fr = f, [], []
                ts.append(t); fr.append(frame)
                total += 1
        flush()
        dst.execute("COMMIT")
        want = src.execute("SELECT count(*) FROM pfp WHERE sensor=?", [s]).fetchone()[0]
        if total != want:
            log(f"  pfp {s}: MISMATCH {total} != {want}. NOT marking done")
            bad.append(s)
            continue
        dst.execute("INSERT INTO done VALUES (?)", [s])
        log(f"  pfp {s}: {total} frames in {time.time()-t0:.0f}s")
    src.close(); _finish(dst)
    if bad:
        log(f"pfp_c.duckdb INCOMPLETE for {', '.join(bad)}; not swapping it in")
        return False
    log("pfp_c.duckdb complete")
    return True


def _payload_rows(path):
    """Total rows across the data tables of one database, or None if unreadable.

    Deliberately counts the DATA tables only (chunks and rows), not psd_lvl or
    the resume bookkeeping: a build file can legitimately carry a pyramid
    forward while holding no captures, and that is exactly the case worth
    catching.
    """
    if not os.path.exists(path):
        return None
    try:
        c = duckdb.connect(path, read_only=True)
    except Exception:                                      # noqa: BLE001
        return None
    try:
        tables = _tables(c)
        total = 0
        for t in ("psd_chunk", "psd", "pfp_chunk", "pfp", "raw", "iq_stft"):
            if t in tables:
                total += c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        return total
    except Exception:                                      # noqa: BLE001
        return None
    finally:
        c.close()


def _keeps_data(built, live):
    """False only when the build file is empty and the live database is not."""
    b, l = _payload_rows(built), _payload_rows(live)
    if b is None or l is None:
        return True            # cannot tell; the per-sensor checks still applied
    return not (b == 0 and l > 0)


def swap_in(name):
    """Move <name>_c.duckdb onto <name>.duckdb, keeping a .bak of the original.

    Doing this here is the point: leaving the compacted file sitting next to
    the original under a different name is how a run ends up "finished" while
    the server still reads the old file.
    """
    built = _built_path(name)
    live = _live_path(name)
    if not os.path.exists(built):
        return False
    # A .wal beside the build file means its contents are NOT all in the .duckdb
    # yet. Renaming just the .duckdb strands them: measured on a nearly-full
    # volume, the live database ended up with zero tables while 2,349 captures
    # sat in an orphaned psd_c.duckdb.wal -- and the run said "DONE ALL", exit 0.
    # _finish's CHECKPOINT should make this unreachable; this is the backstop,
    # and the file's mere presence is an exact discriminator.
    if os.path.exists(built + ".wal"):
        log(f"  REFUSING to swap {name}_c.duckdb in: {name}_c.duckdb.wal still "
            f"exists, so part of the compacted data is not in the file yet.")
        log("    Nothing has been changed. Re-run compact_db.py once there is "
            "free disk space; the build file resumes where it stopped.")
        return False
    # The live database's own .wal has to travel WITH it. Left behind, it either
    # replays into the NEW file and fails ('Table "psd_meta" already exists',
    # database unopenable) or it is discarded and the .bak is a stub holding none
    # of the data it is supposed to be a backup of.
    live_wal = live + ".wal"
    if os.path.exists(live_wal):
        try:
            duckdb.connect(live).close()          # checkpoint it away
        except Exception as e:                                 # noqa: BLE001
            log(f"  REFUSING to swap {name}_c.duckdb in: {name}.duckdb has an "
                f"un-checkpointed .wal that could not be flushed ({e}).")
            log("    Nothing has been changed. Stop any other process holding "
                "the database and re-run.")
            return False
    # Never trade data for no data. The per-sensor row-count checks above verify
    # what was copied, but they say nothing when NOTHING was copied: a build file
    # with zero rows is "complete" by that standard, and swapping it in silently
    # emptied a live database (the empty-rows case in _already_compact). This is
    # the backstop for that whole class of mistake, wherever it comes from.
    if not _keeps_data(built, live):
        log(f"  REFUSING to swap {name}_c.duckdb in: it holds no data and "
            f"{name}.duckdb does. Leaving the live database untouched.")
        log(f"    The build file is kept at {os.path.basename(built)} if you want "
            f"to look at it.")
        return False
    bak = live + ".bak"
    try:
        if os.path.exists(live):
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(live, bak)
            if os.path.exists(live_wal):       # checkpoint above did not remove it
                os.replace(live_wal, bak + ".wal")
        os.replace(built, live)
    except OSError as e:
        log(f"  could not swap {name}_c.duckdb -> {name}.duckdb: {e}")
        log("    Stop serve.py (it holds the file open on Windows) and re-run, "
            "or rename the files by hand.")
        return False
    log(f"  {name}.duckdb <- {name}_c.duckdb "
        f"({os.path.getsize(live)/1e9:.2f} GB; previous kept as "
        f"{os.path.basename(bak)})")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-swap", action="store_true",
                    help="leave the compacted files as *_c.duckdb instead of "
                         "moving them into place")
    args = ap.parse_args()

    def attempt(label, fn, *a):
        """Run one layer's compaction; a failure must not take the others down.

        An ENOSPC or a lock error inside compact_psd used to propagate out of
        __main__, so the run died before psd_median, psd_mean and pfp were even
        looked at -- the same collateral damage _already_compact was written to
        stop for CatalogException. Nothing has been swapped at this point, so a
        failed layer simply stays uncompacted.
        """
        log(f"compacting {label} ...")
        try:
            return fn(*a)
        except Exception as e:                             # noqa: BLE001
            log(f"  {label} FAILED: {e}")
            log(f"    {label} was left exactly as it was; the other layers "
                f"still run. Re-run to retry it.")
            return False

    ok = {"spectrum": attempt("spectrum.duckdb", compact_spectrum)}
    ok["psd"] = attempt("psd.duckdb", compact_psd)
    for extra in ("psd_median", "psd_mean"):
        if os.path.exists(_live_path(extra)):
            ok[extra] = attempt(f"{extra}.duckdb", compact_psd, extra)
    ok["pfp"] = attempt("pfp.duckdb (the big one)", compact_pfp)
    for n in ("spectrum", "psd", "psd_median", "psd_mean", "pfp"):
        p = _built_path(n)             # where this run actually wrote them
        f = os.path.basename(p)
        if os.path.exists(p):
            log(f"  {f}: {os.path.getsize(p)/1e9:.2f} GB")
    if args.no_swap:
        log("--no-swap: rename *_c.duckdb yourself before serve.py will see them")
    else:
        log("swapping compacted files into place ...")
        # Only a verified-complete build file replaces a live database.
        swapped = [n for n in ("spectrum", "psd", "psd_median", "psd_mean", "pfp")
                   if ok.get(n) and swap_in(n)]
        if not swapped:
            log("  nothing to swap")
    log("DONE ALL")
