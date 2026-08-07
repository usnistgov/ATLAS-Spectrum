"""test_repeat.py: prove the ingest can be re-run as new data arrives.

The CBRS datasets grow: a sensor keeps measuring, and next month there is
another folder of exports to fold into databases that already exist. That makes
"run the ingest again" the normal case rather than the exception, and it has to
be *additive* (new days land), *idempotent* (old days are not re-read or
duplicated) and *safe on a compacted database* (the README recommends
compaction, so the second run usually meets the compacted schema, not the one
the first run wrote).

Every stage re-renders a tile through serve.py, because a row count that looks
right while the viewer has stopped drawing is not a pass.

    python examples/test_repeat.py          # exits 0 = PASS

Nothing here touches the real databases: it all happens in a temporary
directory via the SPECTRUM_DB / PSD_DB / PFP_DB / ATLAS_DB_DIR overrides.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ING = os.path.join(ROOT, "ingest")
sys.path.insert(0, HERE)

from test_ingest import (NF, NPOS, SENSOR, CAPS_PER_DAY,  # noqa: E402
                         FRAMES_PER_DAY, HOURS_PER_DAY, CHANNELS,
                         write_batch, run, probe)

WEEK1 = ["2024-01-01", "2024-01-02"]
WEEK2 = ["2024-01-03", "2024-01-04"]
WEEK3 = ["2024-01-05", "2024-01-06"]


failed = 0


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok




def counts(dbs):
    """What is actually stored, by whichever schema is on disk."""
    import duckdb
    out = {}
    for name, tables in (("psd", ("psd", "psd_chunk")),
                         ("pfp", ("pfp", "pfp_chunk"))):
        path = os.path.join(dbs, f"{name}.duckdb")
        out[name] = {}
        if not os.path.exists(path):
            continue
        con = duckdb.connect(path, read_only=True)
        try:
            have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            for t in tables:
                if t not in have:
                    continue
                if t.endswith("_chunk"):
                    out[name][t] = con.execute(
                        f"SELECT COALESCE(SUM(n),0) FROM {t}").fetchone()[0]
                else:
                    out[name][t] = con.execute(
                        f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            # distinct capture instants, the thing duplication would inflate
            if "psd" in have and name == "psd":
                out["psd_distinct_t"] = con.execute(
                    "SELECT COUNT(DISTINCT t) FROM psd").fetchone()[0]
            if "pfp" in have and name == "pfp":
                out["pfp_distinct_t"] = con.execute(
                    "SELECT COUNT(DISTINCT t) FROM pfp").fetchone()[0]
        finally:
            con.close()
    sp = os.path.join(dbs, "spectrum.duckdb")
    if os.path.exists(sp):
        con = duckdb.connect(sp, read_only=True)
        try:
            have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            tbl = "lvl_h1" if "lvl_h1" in have else sorted(have)[0]
            out["summary_rows"] = con.execute(
                f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            out["summary_tmax"] = con.execute(
                f"SELECT MAX(t) FROM {tbl}").fetchone()[0]
        finally:
            con.close()
    return out


def _decode(path, table, payload, width):
    """{t: payload bytes} out of a chunked table, whatever the chunking."""
    import zlib

    import duckdb
    import numpy as np
    out = {}
    con = duckdb.connect(path, read_only=True)
    try:
        rows = con.execute(
            f"SELECT n, times, {payload} FROM {table}").fetchall()
    finally:
        con.close()
    for n, tb, pb in rows:
        ts = np.frombuffer(zlib.decompress(tb), dtype=np.float64)
        raw = zlib.decompress(pb)
        if len(raw) != n * width or len(ts) != n:
            return None
        for i in range(n):
            out[round(float(ts[i]), 6)] = raw[i * width:(i + 1) * width]
    return out


def decoded_match(a_dir, b_dir):
    """Do two compacted databases hold the same captures, byte for byte?"""
    for name, table, payload, width in (
            ("psd", "psd_chunk", "specs", NF),
            ("pfp", "pfp_chunk", "frames", NPOS)):
        A = _decode(os.path.join(a_dir, f"{name}.duckdb"), table, payload, width)
        B = _decode(os.path.join(b_dir, f"{name}.duckdb"), table, payload, width)
        if A is None or B is None:
            return False, f"{name}: a chunk's blob length disagrees with its n"
        if set(A) != set(B):
            return False, (f"{name}: {len(A)} vs {len(B)} capture instants, "
                           f"{len(set(A) ^ set(B))} not in both")
        bad = [t for t in A if A[t] != B[t]]
        if bad:
            return False, f"{name}: {len(bad)} capture(s) differ in content"
    return True, "psd and pfp both identical, instant for instant"


def ingest(dbs, data, env, summaries_dir):
    """One full pass of all three CBRS ingests, as atlas.py would run them."""
    logs = {}
    for script, args in (("psd_ingest.py", [SENSOR, "--root", data]),
                         ("pfp_ingest.py", [SENSOR, "--root", data])):
        rc, out = run([sys.executable, os.path.join(ING, script)] + args, env)
        logs[script] = (rc, out)
    rc, out = run([sys.executable, os.path.join(ING, "build_db.py"),
                   "--csv-dir", summaries_dir], env)
    logs["build_db.py"] = (rc, out)
    return logs


def renders(dbs, tmp, label):
    """The viewer still draws both deep layers."""
    got, out = probe(dbs, tmp)
    if got is None:
        check(f"{label}: serve.py answers at all", False, out[-200:])
        return None
    ok = (got.get("psd_layer", [0])[0] == 200
          and got.get("pfp_frame", [0])[0] == 200)
    check(f"{label}: psd and pfp tiles still render", ok,
          f"psd={got.get('psd_layer')} pfp={got.get('pfp_frame')}")
    return got


def case_midnight_straddle(tmp):
    """An export whose last capture lands just after midnight.

    Every real CBRS export does this -- 2025-07-09_HU_max.csv runs 00:02:39 on
    the 9th through 00:00:22 on the 10th. The resume test used to ask "does the
    database hold any capture on the day this file is NAMED for", so reading day
    D put one capture into day D+1 and marked D+1 complete; D+1's file was then
    skipped whole. Measured on three real days: 2,349 captures became 1,584, a
    third of the data gone, exit code 0.

    Nothing else in this suite has a straddle -- write_batch keeps every capture
    inside its own day -- which is exactly why the bug survived a suite named
    for repeatable ingest.
    """
    import numpy as np
    src = os.path.join(tmp, "straddle_src")
    dbs = os.path.join(tmp, "straddle_dbs")
    os.makedirs(src, exist_ok=True); os.makedirs(dbs, exist_ok=True)
    env = {"PSD_DB": os.path.join(dbs, "psd.duckdb"), "ATLAS_DB_DIR": dbs}
    NF = 2250
    days = ["2024-07-01", "2024-07-02", "2024-07-03"]
    per_day = 5

    def write(day):
        # four captures inside the day, then one just past midnight, the way the
        # real exports end.
        stamps = [f"{day} {h:02d}:10:00.000000+00:00" for h in (0, 6, 12, 18)]
        nxt = str(np.datetime64(day, "D") + 1)
        stamps.append(f"{nxt} 00:00:22.440000+00:00")
        rows = ["sweep_timestamp," + ",".join(str(3530040000.0 + 80000.0 * i)
                                              for i in range(NF))]
        for k, st in enumerate(stamps):
            rows.append(st + "," + ",".join(f"{-140.0 + k * 0.1:.1f}" for _ in range(NF)))
        with open(os.path.join(src, f"{day}_STRAD_max.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")

    def stored():
        return counts(dbs)["psd"].get("psd_chunk") or counts(dbs)["psd"].get("psd") or 0

    write(days[0])
    run([sys.executable, os.path.join(ING, "psd_ingest.py"), "STRAD",
         "--root", src], env)
    run([sys.executable, os.path.join(ING, "compact_db.py")], env)
    first = stored()
    check("a straddling export ingests completely", first == per_day,
          f"{first} captures, expected {per_day}")

    for d in days[1:]:
        write(d)
    run([sys.executable, os.path.join(ING, "psd_ingest.py"), "STRAD",
         "--root", src], env)
    after = stored()
    check("the days after a straddling export are NOT skipped",
          after == per_day * len(days),
          f"{after} captures, expected {per_day * len(days)}")

    # and re-running once more must add nothing at all
    run([sys.executable, os.path.join(ING, "psd_ingest.py"), "STRAD",
         "--root", src], env)
    again = stored()
    check("re-running a straddling set duplicates nothing", again == after,
          f"{after} -> {again} captures")


def main():
    print("repeatable ingest test\n")
    tmp = tempfile.mkdtemp(prefix="atlas-repeat-test-")
    data = os.path.join(tmp, "data")
    dbs = os.path.join(tmp, "dbs")
    os.makedirs(dbs)
    env = {"SPECTRUM_DB": os.path.join(dbs, "spectrum.duckdb"),
           "PSD_DB": os.path.join(dbs, "psd.duckdb"),
           "PFP_DB": os.path.join(dbs, "pfp.duckdb"),
           "ATLAS_DB_DIR": dbs}

    # ---- month 1 ----
    _p, _f, sm = write_batch(data, WEEK1, seed=1, sub="", summaries="Summaries_2024-01-01.csv")
    ingest(dbs, data, env, sm)
    c1 = counts(dbs)
    n1_psd = len(WEEK1) * CAPS_PER_DAY
    n1_pfp = len(WEEK1) * FRAMES_PER_DAY
    check("first run stores every capture",
          c1["psd"].get("psd") == n1_psd and c1["pfp"].get("pfp") == n1_pfp,
          f"psd={c1['psd']} pfp={c1['pfp']} (want {n1_psd}/{n1_pfp})")
    check("first run stores every summary row",
          c1.get("summary_rows") == len(WEEK1) * HOURS_PER_DAY * len(CHANNELS),
          str(c1.get("summary_rows")))
    renders(dbs, tmp, "after month 1")

    # ---- month 2 arrives: the new days land, the old ones are not re-read ----
    _p, _f, sm = write_batch(data, WEEK2, seed=2, sub="", summaries="Summaries_2024-01-03.csv")
    logs = ingest(dbs, data, env, sm)
    c2 = counts(dbs)
    n2_psd = (len(WEEK1) + len(WEEK2)) * CAPS_PER_DAY
    n2_pfp = (len(WEEK1) + len(WEEK2)) * FRAMES_PER_DAY
    check("a second run adds the new days",
          c2["psd"].get("psd") == n2_psd and c2["pfp"].get("pfp") == n2_pfp,
          f"psd={c2['psd']} pfp={c2['pfp']} (want {n2_psd}/{n2_pfp})")
    check("a second run duplicates nothing",
          c2.get("psd_distinct_t") == n2_psd
          and c2.get("pfp_distinct_t") == n2_pfp,
          f"distinct psd t={c2.get('psd_distinct_t')} "
          f"pfp t={c2.get('pfp_distinct_t')}")
    check("the second run says it is skipping the days it already has",
          "already ingested" in logs["psd_ingest.py"][1],
          next((l.strip() for l in logs["psd_ingest.py"][1].splitlines()
                if "already ingested" in l), "no resume line printed"))
    check("the summary layer grew to cover the new days",
          c2.get("summary_rows")
          == (len(WEEK1) + len(WEEK2)) * HOURS_PER_DAY * len(CHANNELS),
          f"{c2.get('summary_rows')} row(s)")
    check("the summary layer's newest timestamp moved forward",
          (c2.get("summary_tmax") or 0) > (c1.get("summary_tmax") or 0),
          f"{c1.get('summary_tmax')} -> {c2.get('summary_tmax')}")
    renders(dbs, tmp, "after month 2")

    # ---- a run with nothing new changes nothing at all ----
    logs = ingest(dbs, data, env, sm)
    c3 = counts(dbs)
    check("a run with no new data is a no-op",
          (c3["psd"].get("psd"), c3["pfp"].get("pfp"), c3.get("summary_rows"))
          == (c2["psd"].get("psd"), c2["pfp"].get("pfp"), c2.get("summary_rows")),
          f"{c2['psd']}/{c2['pfp']}/{c2.get('summary_rows')} -> "
          f"{c3['psd']}/{c3['pfp']}/{c3.get('summary_rows')}")

    # ---- compaction, then month 3: the case the README's advice creates ----
    rc, out = run([sys.executable, os.path.join(ING, "compact_db.py")], env)
    check("compaction succeeds", rc == 0, out.strip()[-160:])
    cc = counts(dbs)
    check("compaction preserved every capture",
          cc["psd"].get("psd_chunk") == n2_psd
          and cc["pfp"].get("pfp_chunk") == n2_pfp,
          f"psd={cc['psd']} pfp={cc['pfp']}")
    renders(dbs, tmp, "after compaction")

    _p, _f, sm = write_batch(data, WEEK3, seed=3, sub="", summaries="Summaries_2024-01-05.csv")
    logs = ingest(dbs, data, env, sm)
    c4 = counts(dbs)
    n3_psd = (len(WEEK1) + len(WEEK2) + len(WEEK3)) * CAPS_PER_DAY
    n3_pfp = (len(WEEK1) + len(WEEK2) + len(WEEK3)) * FRAMES_PER_DAY

    check("ingesting onto a compacted database does not fail",
          logs["psd_ingest.py"][0] == 0 and logs["pfp_ingest.py"][0] == 0
          and "Traceback" not in logs["psd_ingest.py"][1],
          f"psd rc={logs['psd_ingest.py'][0]} "
          f"pfp rc={logs['pfp_ingest.py'][0]}")
    check("the new days land, exactly once, across whatever schemas are present",
          (c4["psd"].get("psd_chunk", 0) + c4["psd"].get("psd", 0)) == n3_psd
          and (c4["pfp"].get("pfp_chunk", 0) + c4["pfp"].get("pfp", 0)) == n3_pfp,
          f"psd chunked={c4['psd'].get('psd_chunk')} "
          f"rows={c4['psd'].get('psd')} (want {n3_psd}); "
          f"pfp chunked={c4['pfp'].get('pfp_chunk')} "
          f"rows={c4['pfp'].get('pfp')} (want {n3_pfp})")
    check("the append stayed in one schema, so serve.py cannot read half of it",
          not (c4["psd"].get("psd_chunk") and c4["psd"].get("psd"))
          and not (c4["pfp"].get("pfp_chunk") and c4["pfp"].get("pfp")),
          f"psd={c4['psd']} pfp={c4['pfp']}")
    got = renders(dbs, tmp, "after month 3 onto a compacted database")
    if got:
        check("the viewer sees the whole history, not just one schema's half",
              got["psd_meta"].get("has") is True
              and got["pfp_meta"].get("has") is True,
              f"psd={got['psd_meta'].get('has')} pfp={got['pfp_meta'].get('has')}")
        check("the newest day is inside the advertised time range",
              (got["psd_meta"].get("t_max") or 0) >= c2.get("summary_tmax", 0),
              f"psd t_max={got['psd_meta'].get('t_max')}")

    # ---- the appended chunks must be byte-identical to compacting instead ----
    # Same six days, built the other way round: all as rows, compacted once at
    # the end. If appending drifts from what compact_db.py writes -- a wrong
    # chunk boundary, a mis-scaled sample, times in the wrong order -- the
    # decoded spectra stop matching and every later render is subtly wrong.
    ref = os.path.join(tmp, "ref")
    os.makedirs(ref)
    renv = {"SPECTRUM_DB": os.path.join(ref, "spectrum.duckdb"),
            "PSD_DB": os.path.join(ref, "psd.duckdb"),
            "PFP_DB": os.path.join(ref, "pfp.duckdb"),
            "ATLAS_DB_DIR": ref}
    ingest(ref, data, renv, os.path.join(data, "summaries"))
    rc, out = run([sys.executable, os.path.join(ING, "compact_db.py")], renv)
    check("the reference database compacts in one pass", rc == 0,
          out.strip()[-120:])
    same, why = decoded_match(dbs, ref)
    check("appending gives the same stored data as compacting in one pass",
          same, why)

    case_midnight_straddle(tmp)

    print()
    if failed:
        print(f"RESULT: FAIL - {failed} check(s) failed")
        return 1
    print("RESULT: PASS - the ingest is additive, idempotent and safe to "
          "repeat on a compacted database")
    return 0


if __name__ == "__main__":
    sys.exit(main())
