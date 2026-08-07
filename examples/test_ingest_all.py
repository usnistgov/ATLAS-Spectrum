"""test_ingest_all.py: the one-command ingest, and the ways it used to lose data.

ingest_all.py runs every layer's ingest and then compaction over a folder it has
never seen. Every check here is a defect that actually happened and was measured,
because this is the script that turns "I added next month's files" into one
command -- and re-running it is the normal case, so a re-run that destroys data
is the worst bug the project can have.

    python examples/test_ingest_all.py        # exits 0 = PASS
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ingest"))
sys.path.insert(0, ROOT)

import duckdb                                              # noqa: E402

failed = 0
NF, NPOS = 2250, 560


def check(name, ok, detail=""):
    global failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failed += 1


def ingest_all(src, dest, *extra):
    env = {**os.environ, "ATLAS_DB_DIR": dest,
           "SPECTRUM_DB": os.path.join(dest, "spectrum.duckdb"),
           "PSD_DB": os.path.join(dest, "psd.duckdb"),
           "PFP_DB": os.path.join(dest, "pfp.duckdb"),
           "IQ_DB": os.path.join(dest, "iq.duckdb")}
    p = subprocess.run([sys.executable, os.path.join(ROOT, "ingest", "ingest_all.py"),
                        src, "--dest", dest, *extra],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=2400)
    return p.returncode, p.stdout + p.stderr


def q(path, sql):
    """One scalar from a database, or None when it or the table is not there."""
    if not os.path.exists(path):
        return None
    c = duckdb.connect(path, read_only=True)
    try:
        return c.execute(sql).fetchone()[0]
    except Exception:                                      # noqa: BLE001
        return None
    finally:
        c.close()


def psd_csv(path, day, n=6, off=0.0, bins=NF):
    rows = ["datetime," + ",".join(f"b{i}" for i in range(bins))]
    rng = np.random.default_rng(abs(hash(path)) % 2**31)
    for k in range(n):
        vals = -140 + off + rng.normal(0, 1.0, bins)
        rows.append(f"{day}T{k*3:02d}:00:00," + ",".join(f"{v:.1f}" for v in vals))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def pfp_csv(path, day, stat_level=0.0, n=4):
    rows = ["datetime,frequency," + ",".join(f"p{i}" for i in range(NPOS))]
    rng = np.random.default_rng(7)
    for k in range(n):
        v = -90 + stat_level + rng.normal(0, 0.5, NPOS)
        rows.append(f"{day}T{k*5:02d}:00:00,3555000000," + ",".join(f"{x:.1f}" for x in v))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def summaries_csv(path, days):
    with open(path, "w") as f:
        f.write("sensor_name,channel_frequency_mhz,timestamp,max,median,mean\n")
        for day in days:
            for hh in range(0, 24, 6):
                for ch in (3555.0, 3625.0):
                    f.write(f"S1,{ch},{day} {hh:02d}:00:00+00,-70.0,-75.0,-80.0\n")


def case_summaries_in_two_directories(tmp):
    """build_db.py REPLACES the summary database wholesale, so one step per
    directory threw away every directory but the last: two months of summaries
    became one, while PSD and PFP correctly accumulated."""
    src, dest = os.path.join(tmp, "sum_src"), os.path.join(tmp, "sum_db")
    for sub, days in (("y2024/summaries", ["2024-01-01", "2024-01-02"]),
                      ("y2025/summaries", ["2025-01-01", "2025-01-02"])):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
        summaries_csv(os.path.join(src, sub, "Summaries_x.csv"), days)
    os.makedirs(dest, exist_ok=True)
    rc, out = ingest_all(src, dest, "--no-compact", "--no-levels")
    db = os.path.join(dest, "spectrum.duckdb")
    years = q(db, "SELECT count(DISTINCT year(to_timestamp(t))) FROM raw")
    check("Summaries in two directories all survive", years == 2,
          f"{years} distinct year(s) in spectrum.duckdb, expected 2")


def case_two_pfp_statistics(tmp):
    """The pfp table has no stat column and no per-stat sibling database, so
    ingesting two statistics interleaved them into one layer whose metadata
    named only one -- a 50 dB step in the middle of the picture."""
    src, dest = os.path.join(tmp, "pfp_src"), os.path.join(tmp, "pfp_db")
    os.makedirs(src, exist_ok=True); os.makedirs(dest, exist_ok=True)
    # DIFFERENT days per statistic. pfp_ingest's resume is per sensor, not per
    # statistic, so a second statistic on the SAME day is skipped and the mixing
    # never happens -- which made an earlier version of this check pass against
    # code that does mix them.
    pfp_csv(os.path.join(src, "PFP_2024-02-01_S1_max_peak.csv"), "2024-02-01", 0.0)
    pfp_csv(os.path.join(src, "PFP_2024-02-02_S1_mean_rms.csv"), "2024-02-02", -50.0)
    rc, out = ingest_all(src, dest, "--no-compact", "--no-levels")
    # Count FRAMES, not pfp_meta statistics: pfp_ingest deletes and re-inserts
    # the meta row per sensor, so the last statistic wins and the distinct count
    # is 1 either way -- while the pfp table quietly holds both. 4 frames per
    # file, so one statistic is 4 and the mixed corruption is 8.
    frames = q(os.path.join(dest, "pfp.duckdb"), "SELECT count(*) FROM pfp")
    check("only one PFP statistic's frames are ingested", frames == 4,
          f"{frames} frame(s) in the pfp table, expected 4 (8 = both mixed)")
    check("the ignored PFP statistic is reported, not silent",
          "ingesting 'max_peak' only" in out or "mean_rms" in out)


def case_rerun_keeps_data(tmp):
    """The measured catastrophe: a second identical run emptied the databases.
    psd_ingest leaves an empty rows table on an already-compacted database, and
    compaction then compacted THAT over the real chunks -- 72 captures to 0."""
    src, dest = os.path.join(tmp, "re_src"), os.path.join(tmp, "re_db")
    os.makedirs(src, exist_ok=True); os.makedirs(dest, exist_ok=True)
    for stat, off in (("max", 0.0), ("median", -10.0)):
        psd_csv(os.path.join(src, f"2024-03-01_S1_{stat}.csv"), "2024-03-01", off=off)
    pfp_csv(os.path.join(src, "PFP_2024-03-01_S1_max_peak.csv"), "2024-03-01")
    rc1, _ = ingest_all(src, dest)
    first = {n: q(os.path.join(dest, f"{n}.duckdb"), sql) for n, sql in (
        ("psd", "SELECT sum(n) FROM psd_chunk"),
        ("psd_median", "SELECT sum(n) FROM psd_chunk"),
        ("pfp", "SELECT sum(n) FROM pfp_chunk"))}
    rc2, out2 = ingest_all(src, dest)
    second = {n: q(os.path.join(dest, f"{n}.duckdb"), sql) for n, sql in (
        ("psd", "SELECT sum(n) FROM psd_chunk"),
        ("psd_median", "SELECT sum(n) FROM psd_chunk"),
        ("pfp", "SELECT sum(n) FROM pfp_chunk"))}
    check("a re-run keeps every capture", first == second and all(first.values()),
          f"{first} -> {second}")
    meta = q(os.path.join(dest, "psd.duckdb"), "SELECT count(*) FROM psd_meta")
    check("a re-run keeps the metadata rows the viewer needs", bool(meta),
          f"psd_meta rows: {meta}")
    return src, dest, first


def case_incremental(tmp, src, dest, before):
    """Next month's file arrives: the databases must GROW, and the pyramid with
    them."""
    for stat, off in (("max", 0.0), ("median", -10.0)):
        psd_csv(os.path.join(src, f"2024-03-02_S1_{stat}.csv"), "2024-03-02", off=off)
    rc, out = ingest_all(src, dest)
    after = q(os.path.join(dest, "psd.duckdb"), "SELECT sum(n) FROM psd_chunk")
    check("a newly arrived day is added", after and after > (before["psd"] or 0),
          f"{before['psd']} -> {after} captures")
    lvl = q(os.path.join(dest, "psd.duckdb"), "SELECT count(*) FROM psd_lvl")
    check("the coarse pyramid covers the new day too", bool(lvl), f"{lvl} level rows")


def case_double_compaction(tmp, dest):
    """compact_spectrum multiplies dBm by 10 into a SMALLINT and had no guard:
    a second run made -85.9 dBm into -859.0, which serve.py cannot detect because
    it reads the scale off the column type. A third run overflowed and aborted
    the whole job."""
    src = os.path.join(tmp, "spec_src")
    os.makedirs(src, exist_ok=True)
    summaries_csv(os.path.join(src, "Summaries_s.csv"), ["2024-04-01"])
    d = os.path.join(tmp, "spec_db"); os.makedirs(d, exist_ok=True)
    ingest_all(src, d, "--no-levels")
    first = q(os.path.join(d, "spectrum.duckdb"), "SELECT min(mx) FROM raw")
    env = {**os.environ, "ATLAS_DB_DIR": d}
    for _ in range(2):
        subprocess.run([sys.executable, os.path.join(ROOT, "ingest", "compact_db.py")],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    again = q(os.path.join(d, "spectrum.duckdb"), "SELECT min(mx) FROM raw")
    check("repeated compaction does not rescale the summary values",
          first is not None and first == again, f"{first} -> {again}")


def case_stale_build_file(tmp, _unused_dest):
    """A leftover *_c.duckdb is a time machine: its `done` table makes
    compaction skip every sensor it already has, call itself complete, and swap
    newer data away. Measured: 800 captures back to 300."""
    # Built fresh here rather than reusing another case's databases: sharing
    # state meant this check was reading whatever the previous cases had left,
    # and it passed against code that has no such protection at all.
    src, dest = os.path.join(tmp, "stale_src"), os.path.join(tmp, "stale_db")
    os.makedirs(src, exist_ok=True); os.makedirs(dest, exist_ok=True)
    psd_csv(os.path.join(src, "2024-06-01_S1_max.csv"), "2024-06-01")
    ingest_all(src, dest, "--no-levels")
    psd_csv(os.path.join(src, "2024-06-02_S1_max.csv"), "2024-06-02")
    ingest_all(src, dest, "--no-levels")          # now compacted AND added to
    live = os.path.join(dest, "psd.duckdb")
    caps = q(live, "SELECT sum(n) FROM psd_chunk")
    stale = os.path.join(dest, "psd_c.duckdb")
    c = duckdb.connect(stale)
    c.execute("CREATE TABLE psd_chunk (sensor VARCHAR, t0 DOUBLE, t1 DOUBLE, "
              "n INT, times BLOB, specs BLOB)")
    c.execute("CREATE TABLE psd_meta (sensor VARCHAR, f0 DOUBLE, df DOUBLE, nf INT, "
              "qmin DOUBLE, qmax DOUBLE, t_min DOUBLE, t_max DOUBLE, captures BIGINT)")
    c.execute("CREATE TABLE done (k VARCHAR)")
    c.execute("INSERT INTO done VALUES ('S1')")   # claims S1 is finished
    c.close()
    env = {**os.environ, "ATLAS_DB_DIR": dest}
    subprocess.run([sys.executable, os.path.join(ROOT, "ingest", "compact_db.py")],
                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    after = q(live, "SELECT sum(n) FROM psd_chunk")
    check("a stale build file cannot roll the live database back",
          after == caps and bool(caps), f"{caps} -> {after} captures")


def case_real_failure_is_reported(tmp):
    """A geometry refusal for one sensor must not license a real failure for
    another: the classification is per sensor and statistic, not per folder."""
    src, dest = os.path.join(tmp, "fail_src"), os.path.join(tmp, "fail_db")
    os.makedirs(src, exist_ok=True); os.makedirs(dest, exist_ok=True)
    psd_csv(os.path.join(src, "2024-05-01_NARROW_max.csv"), "2024-05-01", bins=1024)
    psd_csv(os.path.join(src, "2024-05-01_GOOD_max.csv"), "2024-05-01")
    with open(os.path.join(src, "2024-05-02_GOOD_max.csv"), "w") as f:   # corrupt day
        f.write("datetime," + ",".join(f"b{i}" for i in range(NF)) + "\n")
        f.write("2024-05-02T00:00:00," + ",".join("oops" for _ in range(NF)) + "\n")
    rc, out = ingest_all(src, dest, "--no-compact", "--no-levels")
    check("a real read failure fails the run, even beside a geometry refusal",
          rc != 0, f"exit {rc}")
    check("the sensor with no usable files gets no doomed step",
          "no usable max files for sensor NARROW" in out)


def main():
    print("ingest_all.py: one command, any folder\n")
    tmp = tempfile.mkdtemp(prefix="atlas-ingestall-")
    try:
        case_summaries_in_two_directories(tmp)
        case_two_pfp_statistics(tmp)
        src, dest, before = case_rerun_keeps_data(tmp)
        case_incremental(tmp, src, dest, before)
        case_stale_build_file(tmp, dest)
        case_double_compaction(tmp, dest)
        case_real_failure_is_reported(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - one command ingests any folder, and re-running it is safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
