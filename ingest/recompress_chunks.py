"""recompress_chunks.py: re-zlib an already-compacted DB's chunk blobs at the
new Z level (see chunk_io.py / compact_db.py), for databases compact_db.py has
already turned into psd_chunk / pfp_chunk form.

compact_db.py only ever reads FROM the raw `psd` / `pfp` row tables, which are
gone once a database has been compacted -- so bumping Z there does nothing for
data you already compacted. This script instead decompresses each existing
chunk's blobs and recompresses them at the new level, verifying every one
round-trips to the exact original bytes before it counts as done. spectrum.duckdb
has no zlib in it (it's plain SMALLINT columns) so there is nothing for this
script to do there.

Same shape as compact_db.py: writes a *_z9.duckdb build file next to the
original, tracks finished sensors in a `done` table so a killed run resumes
instead of restarting, and only swaps the build file into place (keeping the
original as <name>.duckdb.bak) once every sensor round-tripped clean.

    py recompress_chunks.py
    py recompress_chunks.py --no-swap
"""
import argparse
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _require  # noqa: F401  -- deps message instead of a traceback

import duckdb

from chunk_io import Z  # the level we're upgrading TO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or ROOT)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# (chunk table, blob columns to recompress, key columns besides sensor)
LAYOUT = {
    "psd": ("psd_chunk", ("times", "specs"), ()),
    "pfp": ("pfp_chunk", ("times", "frames"), ("freq",)),
}


def _missing(src_p):
    if os.path.exists(src_p):
        return False
    log(f"  skip: {os.path.basename(src_p)} not found in {DB_DIR}.")
    return True


def _already_chunked(connection, table):
    have = {r[0] for r in connection.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    return table in have


def recompress(base):
    """-> True only if every sensor round-tripped byte-identical at the new level."""
    table, blob_cols, extra_keys = LAYOUT[base]
    src_p = os.path.join(DB_DIR, f"{base}.duckdb")
    if _missing(src_p):
        return False
    src = duckdb.connect(src_p, read_only=True)
    if not _already_chunked(src, table):
        log(f"  skip: {base}.duckdb has no {table} table yet -- run compact_db.py "
            f"first, this script only re-levels an already-compacted DB.")
        src.close()
        return False

    dst_p = os.path.join(DB_DIR, f"{base}_z9.duckdb")
    dst = duckdb.connect(dst_p)
    dst.execute("CREATE TABLE IF NOT EXISTS done (k VARCHAR)")
    cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})").fetchall()]
    # Mirror the live table's own column list/types instead of hand-writing DDL.
    dst.execute(f"ATTACH '{src_p}' AS s (READ_ONLY)")
    dst.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM s.{table} LIMIT 0")
    dst.execute("DETACH s")

    sensors = [r[0] for r in src.execute(
        f"SELECT DISTINCT sensor FROM {table} ORDER BY 1").fetchall()]
    bad = []
    for s in sensors:
        key = f"{base}:{s}"
        if dst.execute("SELECT 1 FROM done WHERE k=?", [key]).fetchone():
            log(f"  {base} {s}: already done")
            continue
        t0 = time.time()
        dst.execute(f"DELETE FROM {table} WHERE sensor=?", [s])  # partial from a crash
        cur = src.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE sensor=?", [s])
        n_rows = 0
        mismatch = False
        dst.execute("BEGIN")
        placeholders = ", ".join("?" * len(cols))
        insert_sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        while True:
            rows = cur.fetchmany(2048)
            if not rows:
                break
            for row in rows:
                row = list(row)
                for bc in blob_cols:
                    i = cols.index(bc)
                    old_blob = row[i]
                    raw = zlib.decompress(old_blob)
                    new_blob = zlib.compress(raw, Z)
                    if zlib.decompress(new_blob) != raw:   # paranoia: verify before trusting it
                        mismatch = True
                    row[i] = new_blob
                dst.execute(insert_sql, row)
                n_rows += 1
        dst.execute("COMMIT")
        if mismatch:
            log(f"  {base} {s}: ROUND-TRIP MISMATCH. NOT marking done")
            bad.append(s)
            continue
        want = src.execute(f"SELECT count(*) FROM {table} WHERE sensor=?", [s]).fetchone()[0]
        if n_rows != want:
            log(f"  {base} {s}: row count MISMATCH {n_rows} != {want}. NOT marking done")
            bad.append(s)
            continue
        dst.execute("INSERT INTO done VALUES (?)", [key])
        log(f"  {base} {s}: {n_rows} chunk(s) re-leveled in {time.time()-t0:.0f}s")

    # copy over every other table verbatim (meta tables etc.) if not already there.
    # `have` must be read BEFORE attaching src: once attached, information_schema
    # spans both catalogs, so src's own psd_meta/pfp_meta would show up under its
    # bare name and make an unattached dst look like it already has the table --
    # skipping the copy and leaving the rebuilt DB without its meta table.
    other_tables = [r[0] for r in src.execute(
        "SELECT table_name FROM information_schema.tables").fetchall() if r[0] != table]
    have = {r[0] for r in dst.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    dst.execute(f"ATTACH '{src_p}' AS s (READ_ONLY)")
    for t in other_tables:
        if t not in have:
            dst.execute(f"CREATE TABLE {t} AS SELECT * FROM s.{t}")
    dst.execute("DETACH s")

    src.close(); dst.close()
    if bad:
        log(f"{base}_z9.duckdb INCOMPLETE for {', '.join(bad)}; not swapping it in")
        return False
    log(f"{base}_z9.duckdb complete")
    return True


def swap_in(name):
    built = os.path.join(DB_DIR, f"{name}_z9.duckdb")
    live = os.path.join(DB_DIR, f"{name}.duckdb")
    if not os.path.exists(built):
        return False
    # The same backstop compact_db.py carries: never trade data for no data. A
    # build file with zero rows looks "complete" to every per-unit check, and
    # swapping one in silently empties a live database. This function is
    # compact_db.swap_in with the guard missing, which is precisely how it got
    # fixed in one copy and not the other -- so it is imported, not restated.
    sys.path.insert(0, HERE)
    from compact_db import _keeps_data
    if not _keeps_data(built, live):
        log(f"  REFUSING to swap {name}_z9.duckdb in: it holds no data and "
            f"{name}.duckdb does. Leaving the live database untouched.")
        return False
    bak = live + ".bak"
    try:
        if os.path.exists(live):
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(live, bak)
        os.replace(built, live)
    except OSError as e:
        log(f"  could not swap {name}_z9.duckdb -> {name}.duckdb: {e}")
        log("    Stop serve.py (it holds the file open on Windows) and re-run, "
            "or rename the files by hand.")
        return False
    log(f"  {name}.duckdb <- {name}_z9.duckdb "
        f"({os.path.getsize(live)/1e9:.2f} GB; previous kept as "
        f"{os.path.basename(bak)})")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-swap", action="store_true",
                    help="leave the rebuilt files as *_z9.duckdb instead of "
                         "moving them into place")
    args = ap.parse_args()

    ok = {}
    for base in ("psd", "pfp"):
        log(f"recompressing {base}.duckdb chunks to zlib level {Z} ...")
        ok[base] = recompress(base)

    for base in ("psd", "pfp"):
        p = os.path.join(DB_DIR, f"{base}_z9.duckdb")
        if os.path.exists(p):
            log(f"  {base}_z9.duckdb: {os.path.getsize(p)/1e9:.2f} GB")

    if args.no_swap:
        log("--no-swap: rename *_z9.duckdb yourself before serve.py will see them")
    else:
        log("swapping recompressed files into place ...")
        swapped = [n for n in ("psd", "pfp") if ok[n] and swap_in(n)]
        if not swapped:
            log("  nothing to swap")
    log("DONE ALL")
