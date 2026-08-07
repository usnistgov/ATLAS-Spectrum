"""test_durability.py: the ways a re-run used to lose or corrupt data.

Every case here is a measured failure, not a hypothetical. They share one shape:
the exit code said success and the data was wrong, which is the only kind of
failure that survives to production. So each test asserts on the DATA -- capture
counts against the source CSV, duplicate timestamps, whether a database is still
openable -- and never on the exit code alone.

    py examples/test_durability.py                # all cases
    py examples/test_durability.py --case partial_day

Every case generates its own fixture, so this needs no download and no real
exports. That is deliberate: a fixture can be made to straddle midnight, hold a
blank cell, or carry a +05:45 offset on demand, and the real exports cannot.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ING = os.path.join(ROOT, "ingest")
sys.path.insert(0, HERE)
sys.path.insert(0, ING)

import duckdb                                            # noqa: E402

NF = 2250
F0, DF = 3530040000.0, 80000.0
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""),
          flush=True)
    return ok


def write_csv(path, day, sensor, stat, n, seed=1, offset="+00:00", straddle=True):
    """One export. `straddle` makes it run past the next midnight, as real ones do."""
    rng = np.random.default_rng(seed)
    y, m, d = (int(x) for x in day.split("-"))
    base = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T00:02:19", "us").astype("int64")
    span = (86400 + 80) * 1_000_000 if straddle else 86000 * 1_000_000
    step = span // max(n - 1, 1)
    with open(path, "w", newline="") as fh:
        fh.write("sweep_timestamp,"
                 + ",".join(f"{F0 + i * DF:.1f}" for i in range(NF)) + "\n")
        for k in range(n):
            us = int(base + k * step)
            stamp = str(np.datetime64(us, "us")).replace("T", " ") + offset
            v = np.round(-150 + 30 * rng.random(NF), 1)
            fh.write(stamp + "," + ",".join(f"{x:.1f}" for x in v) + "\n")
    return path


def csv_rows(path):
    with open(path) as fh:
        return sum(1 for _ in fh) - 1


def run(args, env=None, cwd=ROOT, timeout=600):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([sys.executable] + args, cwd=cwd, env=e,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def stored(db, sensor="FIX"):
    """(captures, distinct instants) however the database is shaped."""
    if not os.path.exists(db):
        return 0, 0
    c = duckdb.connect(db, read_only=True)
    try:
        have = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        ts = []
        if "psd_chunk" in have:
            for (blob,) in c.execute("SELECT times FROM psd_chunk WHERE sensor=?",
                                     [sensor]).fetchall():
                ts += np.frombuffer(zlib.decompress(blob), np.float64).tolist()
        if "psd" in have:
            ts += [r[0] for r in c.execute("SELECT t FROM psd WHERE sensor=?",
                                           [sensor]).fetchall()]
        return len(ts), len({int(round(t * 1000)) for t in ts})
    finally:
        c.close()


def openable(db):
    try:
        c = duckdb.connect(db, read_only=True)
        c.execute("SELECT 1").fetchone()
        c.close()
        return True
    except Exception:                                      # noqa: BLE001
        return False


# ---------------------------------------------------------------- the cases

def case_partial_day(tmp):
    """A day killed part-way must not be recorded as done.

    Measured before the per-day transaction: a SIGKILL 7 s into a three-day
    ingest left 1,637 of 2,349 captures, and the next run printed "3 of 3 day(s)
    already ingested", exit code 0. The gap was permanent -- the resume test asks
    whether a day was STARTED, and hundreds of committed captures answer yes.
    """
    src = os.path.join(tmp, "pd", "src")
    os.makedirs(src)
    days = ["2025-07-09", "2025-07-10", "2025-07-11"]
    want = 0
    for i, d in enumerate(days):
        want += csv_rows(write_csv(os.path.join(src, f"{d}_FIX_max.csv"),
                                   d, "FIX", "max", 300, seed=i))
    db_dir = os.path.join(tmp, "pd")
    env = {"ATLAS_DB_DIR": db_dir}
    db = os.path.join(db_dir, "psd.duckdb")

    # Kill it mid-run, at several moments, so the kill lands in different places.
    killed_at_least_once = False
    for delay in (1.2, 2.0, 2.8):
        p = subprocess.Popen(
            [sys.executable, os.path.join(ING, "psd_ingest.py"), "FIX",
             "--stat", "max", "--root", src],
            cwd=ROOT, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(delay)
        if p.poll() is None:
            p.send_signal(signal.SIGKILL)
            killed_at_least_once = True
        p.wait(timeout=60)
        n, distinct = stored(db)
        if n and n < want:
            break
    check("a killed ingest leaves whole days only, never a partial one",
          stored(db)[0] % 300 == 0 or stored(db)[0] == 0,
          f"{stored(db)[0]} captures stored; a day is 300")

    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", src], env)
    n, distinct = stored(db)
    check("re-running after a kill recovers every capture",
          n == want and distinct == want, f"{n} stored ({distinct} distinct) "
          f"of {want} in the csvs; killed mid-run: {killed_at_least_once}")
    check("recovery introduces no duplicates", n == distinct,
          f"{n} stored, {distinct} distinct")


def case_interrupt_stops(tmp):
    """Ctrl+C must stop the run, not be counted as one file's error.

    DuckDB turns SIGINT into InterruptException, an ordinary Exception, so the
    per-file handler swallowed it and carried on to the next day.
    """
    src = os.path.join(tmp, "int", "src")
    os.makedirs(src)
    for i, d in enumerate(["2025-07-09", "2025-07-10", "2025-07-11"]):
        write_csv(os.path.join(src, f"{d}_FIX_max.csv"), d, "FIX", "max", 400,
                  seed=i)
    env = {"ATLAS_DB_DIR": os.path.join(tmp, "int")}
    p = subprocess.Popen(
        [sys.executable, os.path.join(ING, "psd_ingest.py"), "FIX", "--stat",
         "max", "--root", src],
        cwd=ROOT, env={**os.environ, **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    p.send_signal(signal.SIGINT)
    try:
        out = p.communicate(timeout=60)[0]
    except subprocess.TimeoutExpired:
        p.kill()
        out = p.communicate()[0]
    check("Ctrl+C stops the ingest instead of skipping one day",
          p.returncode != 0 and "interrupted" in out.lower(),
          f"exit {p.returncode}")
    db = os.path.join(tmp, "int", "psd.duckdb")
    n, distinct = stored(db)
    check("an interrupted ingest stores whole days only",
          n % 400 == 0 and n == distinct, f"{n} stored, {distinct} distinct")


def case_blank_cell(tmp):
    """A blank value cell must be refused, not stored as full-scale power.

    It used to arrive as a masked array whose fill was 0.0 dBm/Hz, which clips to
    byte 255 -- the top of the colour scale, against a noise floor near -152.
    """
    src = os.path.join(tmp, "blank", "src")
    os.makedirs(src)
    p = write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09",
                  "FIX", "max", 20)
    lines = open(p).read().splitlines()
    parts = lines[3].split(",")
    parts[700] = ""
    lines[3] = ",".join(parts)
    open(p, "w").write("\n".join(lines) + "\n")
    env = {"ATLAS_DB_DIR": os.path.join(tmp, "blank")}
    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", src], env)
    check("a blank value cell is refused with a reason",
          rc != 0 and "blank" in out.lower() and "Traceback" not in out,
          f"exit {rc}")
    check("nothing is stored for a file with a blank cell",
          stored(os.path.join(tmp, "blank", "psd.duckdb"))[0] == 0)


def case_blank_timestamp(tmp):
    """A blank timestamp must be refused: it used to store t = NaN.

    psd_meta's range then became NaN, which is not valid JSON, so /api/psd_meta
    was unparseable in the browser and the PSD layer never loaded -- with no
    message anywhere.
    """
    src = os.path.join(tmp, "bts", "src")
    os.makedirs(src)
    p = write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09",
                  "FIX", "max", 20)
    lines = open(p).read().splitlines()
    lines[5] = "," + lines[5].split(",", 1)[1]
    open(p, "w").write("\n".join(lines) + "\n")
    env = {"ATLAS_DB_DIR": os.path.join(tmp, "bts")}
    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", src], env)
    check("a blank timestamp is refused with a reason",
          rc != 0 and "timestamp" in out.lower() and "Traceback" not in out,
          f"exit {rc}")


def case_empty_export(tmp):
    """A header-only export must not count as a day done.

    On a compacted database it printed "caps=0 done=1 err=0" and exited 0, so the
    real file for that day could never be picked up.
    """
    d = os.path.join(tmp, "empty")
    src = os.path.join(d, "src")
    os.makedirs(src)
    # Deliberately NOT straddling: a straddling day 09 would leave a capture on
    # day 10, and the resume check would then (correctly) read the day-10 file's
    # own first capture, find none, and skip it -- masking what this case tests.
    write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09", "FIX",
              "max", 30, straddle=False)
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    run([os.path.join(ING, "compact_db.py")], env)       # now chunk-shaped
    src2 = os.path.join(d, "src2")
    os.makedirs(src2)
    hdr = open(os.path.join(src, "2025-07-09_FIX_max.csv")).readline()
    open(os.path.join(src2, "2025-07-10_FIX_max.csv"), "w").write(hdr)
    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", src2], env)
    check("a header-only export on a compacted database is an error",
          rc != 0 and "no data rows" in out, f"exit {rc}: {out.splitlines()[-1][:70]}")


def case_offset_resume(tmp):
    """An export stamped with a non-UTC offset must not re-ingest on resume.

    first_capture_time stripped the offset and read the digits as UTC while the
    ingest stored the real instant, so the resume check looked for a capture that
    was never stored: 3 rows became 6, exit 0, silently double-weighting every
    mean and median window.
    """
    src = os.path.join(tmp, "off", "src")
    os.makedirs(src)
    p = write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09",
                  "FIX", "max", 12, offset="+05:45", straddle=False)
    env = {"ATLAS_DB_DIR": os.path.join(tmp, "off")}
    db = os.path.join(tmp, "off", "psd.duckdb")
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    first = stored(db)
    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", src], env)
    n, distinct = stored(db)
    check("an offset-stamped export is not re-ingested on resume",
          n == first[0] and n == distinct,
          f"{first[0]} after one run, {n} after two ({distinct} distinct)")
    import cbrs_files
    t = cbrs_files.first_capture_time(p)
    c = duckdb.connect(db, read_only=True)
    try:
        have = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        if "psd_chunk" in have and c.execute(
                "SELECT count(*) FROM psd_chunk").fetchone()[0]:
            blob = c.execute("SELECT times FROM psd_chunk ORDER BY t0 LIMIT 1"
                             ).fetchone()[0]
            t0 = float(np.frombuffer(zlib.decompress(blob), np.float64)[0])
        else:
            t0 = c.execute("SELECT min(t) FROM psd").fetchone()[0]
    finally:
        c.close()
    check("first_capture_time honours the UTC offset",
          t is not None and abs(t - t0) < 0.002,
          f"resume test says {t}, the database stores {t0}")


def case_collision(tmp):
    """Two files claiming one sensor-day must be reported, and resolved stably."""
    d = os.path.join(tmp, "coll")
    a, b = os.path.join(d, "A"), os.path.join(d, "B")
    os.makedirs(a)
    os.makedirs(b)
    write_csv(os.path.join(a, "2025-07-09_FIX_max.csv"), "2025-07-09", "FIX",
              "max", 20, seed=1)
    write_csv(os.path.join(b, "2025-07-09_FIX_max.csv"), "2025-07-09", "FIX",
              "max", 20, seed=99)
    env = {"ATLAS_DB_DIR": d}
    rc, out = run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
                   "--root", d], env)
    check("a duplicated sensor-day is reported, not silently picked",
          "claimed by more than one file" in out, out.splitlines()[0][:70])
    import re as _re
    import cbrs_files
    NAME_RE = _re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})_(?P<sensor>.+?)_"
                          r"(?P<stat>[A-Za-z0-9]+)\.csv$")
    picks = {cbrs_files.discover(d, NAME_RE, "max")["FIX"]["2025-07-09"]
             for _ in range(5)}
    check("the winner is deterministic", len(picks) == 1, str(picks))


def case_swap_wal(tmp):
    """A .wal beside the build file must block the swap.

    On a nearly-full volume DuckDB could not checkpoint on close and did not
    raise, so compaction reported success and swap_in renamed only the .duckdb --
    leaving the live database with ZERO tables and the data stranded in an
    orphaned psd_c.duckdb.wal. Exit code 0, "DONE ALL".
    """
    d = os.path.join(tmp, "wal")
    src = os.path.join(d, "src")
    os.makedirs(src)
    want = csv_rows(write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"),
                              "2025-07-09", "FIX", "max", 40))
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    rc, out = run([os.path.join(ING, "compact_db.py"), "--no-swap"], env)
    built = os.path.join(d, "psd_c.duckdb")
    check("the build file exists to swap", os.path.exists(built))
    open(built + ".wal", "wb").write(b"not a real wal")
    sys.path.insert(0, ING)
    import importlib
    import compact_db
    os.environ["ATLAS_DB_DIR"] = d
    importlib.reload(compact_db)
    swapped = compact_db.swap_in("psd")
    live = os.path.join(d, "psd.duckdb")
    check("a build file with a leftover .wal is not swapped in", swapped is False)
    check("the live database is untouched and still holds its captures",
          openable(live) and stored(live)[0] == want,
          f"{stored(live)[0]} of {want}")
    os.remove(built + ".wal")


def case_live_wal(tmp):
    """The live database's own .wal must travel with it, not be left behind.

    Left beside the NEW file it replays into it and the database becomes
    unopenable ('Table "psd_meta" already exists'); and the .bak it was supposed
    to belong to is a stub holding none of the data.
    """
    d = os.path.join(tmp, "lwal")
    src = os.path.join(d, "src")
    os.makedirs(src)
    want = csv_rows(write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"),
                              "2025-07-09", "FIX", "max", 40))
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    live = os.path.join(d, "psd.duckdb")
    open(live + ".wal", "wb").write(b"")          # an empty, harmless wal
    rc, out = run([os.path.join(ING, "compact_db.py")], env)
    check("compaction with a live .wal leaves an openable database",
          openable(live), out.splitlines()[-1][:70] if out else "")
    check("and it still holds every capture", stored(live)[0] == want,
          f"{stored(live)[0]} of {want}")
    check("no orphan .wal is left beside the new database",
          not os.path.exists(live + ".wal"))


def case_stale_build(tmp):
    """A build file from an older, smaller source must not roll data back.

    Interrupt compaction, ingest another day, compact again: the old build file's
    `done` table made the second run skip every sensor it already had, call
    itself complete, and swap the new day away. Measured: 21,904 captures to
    20,348, exit code 0.
    """
    d = os.path.join(tmp, "stale")
    src = os.path.join(d, "src")
    os.makedirs(src)
    n1 = csv_rows(write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"),
                            "2025-07-09", "FIX", "max", 40, seed=1))
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    run([os.path.join(ING, "compact_db.py"), "--no-swap"], env)   # build file made
    built = os.path.join(d, "psd_c.duckdb")
    check("a build file was produced", os.path.exists(built))
    # the normal monthly cycle: another day into the still-row-shaped live db
    n2 = csv_rows(write_csv(os.path.join(src, "2025-07-11_FIX_max.csv"),
                            "2025-07-11", "FIX", "max", 40, seed=2))
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    live = os.path.join(d, "psd.duckdb")
    before = stored(live)[0]
    rc, out = run([os.path.join(ING, "compact_db.py")], env)
    after = stored(live)[0]
    check("a stale build file does not roll the live database back",
          after >= before, f"{before} captures before compaction, {after} after")
    check("and the live database still holds both days in full",
          after == n1 + n2, f"{after} captures stored, {n1} + {n2} in the csvs")
    check("the stale build file was recognised as stale",
          "stale" in out.lower(), out.splitlines()[0][:70] if out else "")


def case_double_compact(tmp):
    """Compacting twice must be a no-op, byte for byte."""
    d = os.path.join(tmp, "twice")
    src = os.path.join(d, "src")
    os.makedirs(src)
    want = csv_rows(write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"),
                              "2025-07-09", "FIX", "max", 40))
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    run([os.path.join(ING, "compact_db.py")], env)
    live = os.path.join(d, "psd.duckdb")
    first = open(live, "rb").read()
    rc, out = run([os.path.join(ING, "compact_db.py")], env)
    check("a second compaction changes nothing",
          open(live, "rb").read() == first and stored(live)[0] == want,
          f"{stored(live)[0]} of {want}")


def case_mixed_shape_served(tmp):
    """A database holding BOTH shapes must not be served as if the chunks were all.

    serve.py picked the shape by table PRESENCE, so an empty psd_chunk beside a
    full row table advertised the layer and 404'd every tile, and a database with
    data in both served only the chunked part with gap=false and nothing said so.
    """
    d = os.path.join(tmp, "mixed")
    src = os.path.join(d, "src")
    os.makedirs(src)
    write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09", "FIX",
              "max", 30)
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    live = os.path.join(d, "psd.duckdb")
    c = duckdb.connect(live)
    c.execute("""CREATE TABLE IF NOT EXISTS psd_chunk (sensor VARCHAR, t0 DOUBLE,
                 t1 DOUBLE, n INT, times BLOB, specs BLOB)""")   # empty
    c.close()
    probe = os.path.join(d, "probe.py")
    open(probe, "w").write(
        "import json, os, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import serve\n"
        "c = serve.app.test_client()\n"
        "print('PROBE ' + json.dumps({'meta': c.get('/api/psd_meta?sensor=FIX')"
        ".get_json()}))\n")
    rc, out = run([probe], {"ATLAS_DB_DIR": d,
                            "SPECTRUM_DB": os.path.join(d, "none.duckdb"),
                            "PFP_DB": os.path.join(d, "none.duckdb"),
                            "IQ_DB": os.path.join(d, "none.duckdb")})
    line = next((l for l in out.splitlines() if l.startswith("PROBE ")), "")
    import json as _json
    got = _json.loads(line[6:])["meta"] if line else {}
    check("an empty chunk table beside real rows still serves the rows",
          got.get("has") is True, str(got)[:120])


def case_corrupt_summary(tmp):
    """A corrupt spectrum.duckdb must not take the whole server down.

    That open was at module scope with no try, so a truncated or non-DuckDB
    summary file killed the process before the port was bound -- taking the PSD,
    PFP and IQ layers with it, none of which had anything wrong.
    """
    d = os.path.join(tmp, "corrupt")
    src = os.path.join(d, "src")
    os.makedirs(src)
    write_csv(os.path.join(src, "2025-07-09_FIX_max.csv"), "2025-07-09", "FIX",
              "max", 30)
    env = {"ATLAS_DB_DIR": d}
    run([os.path.join(ING, "psd_ingest.py"), "FIX", "--stat", "max",
         "--root", src], env)
    with open(os.path.join(d, "spectrum.duckdb"), "wb") as fh:
        fh.write(os.urandom(200_000))
    probe = os.path.join(d, "probe.py")
    open(probe, "w").write(
        "import json, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import serve\n"
        "c = serve.app.test_client()\n"
        "print('PROBE ' + json.dumps({'meta': c.get('/api/psd_meta?sensor=FIX')"
        ".get_json()}))\n")
    rc, out = run([probe], {"ATLAS_DB_DIR": d,
                            "PFP_DB": os.path.join(d, "none.duckdb"),
                            "IQ_DB": os.path.join(d, "none.duckdb")})
    line = next((l for l in out.splitlines() if l.startswith("PROBE ")), "")
    import json as _json
    got = _json.loads(line[6:])["meta"] if line else {}
    check("a corrupt spectrum.duckdb still serves the PSD layer",
          got.get("has") is True, f"exit {rc}; {str(got)[:80]}")
    check("and it says why the summary layer is missing",
          "could not be read" in out, out.splitlines()[0][:80] if out else "")


def case_summary_rebuild(tmp):
    """A failed summaries rebuild must not destroy the working summary layer.

    build_db.py deleted spectrum.duckdb and then rebuilt it, so from the moment
    the step started there was no zoomed-out layer -- and if the run died, a
    12 KB stub was all that was left of a 0.5 GB database. Observed live on a real
    install mid-run.
    """
    d = os.path.join(tmp, "sum")
    csvd = os.path.join(d, "csv")
    os.makedirs(csvd)
    # A minimal Summaries export: the columns build_db.py needs.
    rows = ["sensor_name,channel_frequency_mhz,timestamp,max,median,mean"]
    for k in range(50):
        us = int(np.datetime64("2025-07-09T00:02:19", "us").astype("int64")
                 + k * 60_000_000)
        stamp = str(np.datetime64(us, "us")).replace("T", " ") + "+00:00"
        rows.append(f"FIX,3550.0,{stamp},"
                    f"{-80 - k * 0.1:.1f},{-95 - k * 0.1:.1f},{-105 - k * 0.1:.1f}")
    open(os.path.join(csvd, "2025_07.csv"), "w").write("\n".join(rows) + "\n")
    env = {"ATLAS_DB_DIR": d}
    rc, out = run([os.path.join(ING, "build_db.py"), "--csv-dir", csvd], env)
    live = os.path.join(d, "spectrum.duckdb")
    check("a summaries build produces a database", rc == 0 and openable(live),
          f"exit {rc}")
    good = os.path.getsize(live)

    # Now point it at a directory of NOTHING it can use, which is what a failed
    # or interrupted rebuild looks like from the live database's point of view.
    empty = os.path.join(d, "bad")
    os.makedirs(empty)
    open(os.path.join(empty, "2025_08.csv"), "w").write("not,a,summaries,export\n1,2,3,4\n")
    rc, out = run([os.path.join(ING, "build_db.py"), "--csv-dir", empty], env)
    check("a failed summaries rebuild leaves the previous database intact",
          openable(live) and os.path.getsize(live) == good,
          f"exit {rc}; {os.path.getsize(live)} bytes vs {good} before")

    # And a killed rebuild, mid-flight.
    p = subprocess.Popen(
        [sys.executable, os.path.join(ING, "build_db.py"), "--csv-dir", csvd],
        cwd=ROOT, env={**os.environ, **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.6)
    if p.poll() is None:
        p.send_signal(signal.SIGKILL)
    p.wait(timeout=60)
    check("a killed summaries rebuild leaves the previous database intact",
          openable(live) and os.path.getsize(live) == good,
          f"{os.path.getsize(live)} bytes vs {good} before")

    # A backup is worth keeping only if it holds something. An earlier failed run
    # can leave a stub here, and backing THAT up litters the folder with a file
    # whose name promises a database and delivers nothing.
    stub = os.path.join(d, "spectrum.duckdb")
    os.replace(live, os.path.join(d, "keep.duckdb"))
    for leftover in (live + ".bak", live + ".bak.wal"):
        if os.path.exists(leftover):      # a killed run above may have swapped
            os.remove(leftover)
    import duckdb as _dd
    _dd.connect(stub).close()                       # a 12 KB empty database
    rc, out = run([os.path.join(ING, "build_db.py"), "--csv-dir", csvd], env)
    check("replacing an EMPTY summary database keeps no backup",
          rc == 0 and not os.path.exists(live + ".bak")
          and "no backup kept" in out,
          f"exit {rc}; .bak exists: {os.path.exists(live + '.bak')}")


def case_summaries_scan_cost(tmp):
    """build_db.py must not OPEN a file to learn it is not a Summaries export.

    It used to hand every .csv under the scan root to DuckDB to discover its
    columns. On an on-demand filesystem every open is a download, and the real
    CBRS tree holds ~44,600 PSD/PFP exports against 24 summaries files -- which
    sort first, so all of them were downloaded before the first usable file was
    reached. Observed live: over an hour, zero bytes written, a 12 KB database,
    indistinguishable from a hang.
    """
    d = os.path.join(tmp, "scan")
    os.makedirs(os.path.join(d, "PSD"))
    os.makedirs(os.path.join(d, "Summaries"))
    hdr = ("sweep_timestamp,"
           + ",".join(f"{F0 + i * DF:.1f}" for i in range(NF)))
    for k in range(30):
        open(os.path.join(d, "PSD", f"2025-07-{k % 28 + 1:02d}_GMM_max.csv"),
             "w").write(hdr + "\n")
    rows = ["sensor_name,channel_frequency_mhz,timestamp,max,median,mean"]
    for k in range(40):
        us = int(np.datetime64("2025-07-09T00:02:19", "us").astype("int64")
                 + k * 60_000_000)
        stamp = str(np.datetime64(us, "us")).replace("T", " ") + "+00:00"
        rows.append(f"FIX,3550.0,{stamp},{-80 - k * 0.1:.1f},"
                    f"{-95 - k * 0.1:.1f},{-105 - k * 0.1:.1f}")
    open(os.path.join(d, "Summaries", "2025_07.csv"), "w").write(
        "\n".join(rows) + "\n")

    # Count the CSVs actually opened while classifying the tree.
    probe = os.path.join(d, "probe.py")
    open(probe, "w").write(
        "import builtins, json, os, sys\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'ingest')!r})\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "opened = []\n"
        "real = builtins.open\n"
        "def spy(f, *a, **k):\n"
        "    if isinstance(f, str) and f.endswith('.csv'):\n"
        "        opened.append(os.path.basename(f))\n"
        "    return real(f, *a, **k)\n"
        "builtins.open = spy\n"
        "import build_db\n"
        f"files, other = build_db.find_csvs({d!r})\n"
        "builtins.open = real\n"
        "print('PROBE ' + json.dumps({'keep': len(files), 'other': len(other),\n"
        "                             'opened': opened}))\n")
    rc, out = run([probe], {"ATLAS_DB_DIR": d})
    line = next((l for l in out.splitlines() if l.startswith("PROBE ")), "")
    import json as _json
    got = _json.loads(line[6:]) if line else {}
    check("build_db finds the summaries file among the decoys",
          got.get("keep") == 1 and got.get("other", 0) >= 28, str(got)[:90])
    check("build_db opens ONLY the summaries candidates, not every csv",
          got.get("opened") == ["2025_07.csv"],
          f"{len(got.get('opened', []))} csv(s) opened: "
          f"{got.get('opened', [])[:4]}")

    # And it still builds correctly when pointed at the whole tree.
    rc, out = run([os.path.join(ING, "build_db.py"), "--csv-dir", d],
                  {"ATLAS_DB_DIR": d})
    check("build_db still builds when pointed at a whole dataset root",
          rc == 0 and openable(os.path.join(d, "spectrum.duckdb")),
          f"exit {rc}")


CASES = {
    "partial_day": case_partial_day,
    "summaries_scan_cost": case_summaries_scan_cost,
    "summary_rebuild": case_summary_rebuild,
    "interrupt_stops": case_interrupt_stops,
    "blank_cell": case_blank_cell,
    "blank_timestamp": case_blank_timestamp,
    "empty_export": case_empty_export,
    "offset_resume": case_offset_resume,
    "collision": case_collision,
    "swap_wal": case_swap_wal,
    "live_wal": case_live_wal,
    "stale_build": case_stale_build,
    "double_compact": case_double_compact,
    "mixed_shape_served": case_mixed_shape_served,
    "corrupt_summary": case_corrupt_summary,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", nargs="*", default=None, choices=list(CASES))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="durability-")
    t0 = time.time()
    try:
        for name in (args.case or CASES):
            print(f"\n-- {name}", flush=True)
            try:
                CASES[name](tmp)
            except Exception as e:                         # noqa: BLE001
                import traceback
                check(f"{name} ran", False, f"{type(e).__name__}: {e}")
                traceback.print_exc()
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed in {time.time()-t0:.0f}s")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
