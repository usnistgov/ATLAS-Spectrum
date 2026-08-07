"""test_cli.py: every input `python atlas.py` accepts, exercised.

The other suites check that each feature works when called correctly. This one
walks the surface a person actually touches: every subcommand, every flag, the
menu in each of the four states it can be in, and *every number in every menu*
-- including the nested "which dataset" and "how much of it" menus, which
download and ingest for real against the local repository fixture in
test_fetch.py.

A menu entry that would block or leave the machine is stubbed and nothing else:
`serve` would never return, and the dataset menu is pointed at the local
fixture instead of data.nist.gov. Everything else runs for real.

    python examples/test_cli.py          # exits 0 = PASS
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "atlas.py")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import test_fetch as TF                                       # noqa: E402
import test_ingest as TI                                      # noqa: E402

failed = 0

# Driven in-process so the menu can be answered and `serve` stubbed. Takes the
# folder default_roots() should report, then one answer per prompt.
HARNESS = '''
import builtins, sys
sys.path.insert(0, %(root)r)
import atlas

root, answers = sys.argv[1], list(sys.argv[2:])
atlas.interactive = lambda: True            # stdout is a pipe here
atlas.default_roots = lambda: [] if root == "-" else [root]
atlas.deep_roots = atlas.default_roots      # never search the real machine
atlas.cmd_serve = lambda args: (print("[stub] viewer would start now"), 0)[1]


def answer(prompt=""):
    print(prompt, end="")
    if not answers:
        raise EOFError
    a = answers.pop(0)
    print(a)
    return a


builtins.input = answer
sys.exit(atlas.cmd_auto(atlas.ns()) or 0)
'''


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def env_for(dbs, extra=None):
    return {**os.environ,
            "SPECTRUM_DB": os.path.join(dbs, "spectrum.duckdb"),
            "PSD_DB": os.path.join(dbs, "psd.duckdb"),
            "PFP_DB": os.path.join(dbs, "pfp.duckdb"),
            "IQ_DB": os.path.join(dbs, "iq.duckdb"),
            "ATLAS_DB_DIR": dbs, **(extra or {})}


def cli(args, dbs, extra=None, timeout=900):
    p = subprocess.run([sys.executable, ATLAS] + args, env=env_for(dbs, extra),
                       cwd=ROOT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def menu(harness, root, answers, dbs, extra=None, timeout=900):
    p = subprocess.run([sys.executable, harness, root] + list(answers),
                       env=env_for(dbs, extra), cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def items_of(out):
    """The numbered choices a menu printed, in order."""
    got = []
    for line in out.splitlines():
        s = line.strip()
        if len(s) > 2 and s[0].isdigit() and s[1:3] == ") ":
            got.append(s[3:].split("      <-")[0].strip())
    return got


def main():
    print("atlas.py full-surface CLI and menu test\n")
    tmp = tempfile.mkdtemp(prefix="atlas-cli-surface-")
    srv = None
    try:
        harness = os.path.join(tmp, "harness.py")
        with open(harness, "w") as f:
            f.write(HARNESS % {"root": ROOT})

        data = os.path.join(tmp, "data")
        TI.make_data(data)
        empty = os.path.join(tmp, "nodata")
        os.makedirs(empty)

        # ---- 1. the command surface ----
        dbs = os.path.join(tmp, "dbs-help")
        os.makedirs(dbs)
        rc, out = cli(["--help"], dbs)
        check("--help lists every subcommand",
              rc == 0 and all(c in out for c in
                              ("doctor", "setup", "status", "scan", "serve",
                               "get")), "")
        subs = ["doctor", "setup", "status", "scan", "serve", "get"]
        bad = []
        for s in subs:
            rc, out = cli([s, "--help"], dbs)
            if rc != 0 or "usage:" not in out.lower():
                bad.append(f"{s} (rc={rc})")
        check(f"every subcommand has working --help ({len(subs)} of them)",
              not bad, "; ".join(bad))

        rc, out = cli(["get"], dbs)
        check("get with no argument is a clean usage error",
              rc != 0 and "Traceback" not in out
              and ("required" in out or "usage" in out.lower()))

        rc, out = cli(["no-such-command"], dbs)
        check("an unknown subcommand is refused cleanly",
              rc != 0 and "Traceback" not in out)

        bad = []
        for s in subs:
            rc, out = cli([s, "--definitely-not-a-flag"], dbs)
            if s == "get":                 # get forwards unknown flags on
                continue
            if rc == 0 or "unrecognized arguments" not in out:
                bad.append(f"{s} (rc={rc})")
        check("a typo'd flag is rejected on every subcommand but get",
              not bad, "; ".join(bad))

        rc, out = cli(["status"], dbs)
        check("status works with nothing built",
              rc == 0 and "Nothing is built yet" in out)
        rc, out = cli(["scan", empty], dbs)
        check("scan of an empty folder exits non-zero and says so",
              rc != 0 and "Found nothing to ingest" in out)
        rc, out = cli(["doctor", "--dry-run"], dbs)
        check("doctor --dry-run is safe on a bare repo",
              rc == 0 and "nothing will change" in out)
        rc, out = cli(["serve", "--dry-run"], dbs)
        check("serve --dry-run prints the command without starting it",
              rc == 0 and "serve.py" in out)
        rc, out = cli(["setup", "--dry-run"], dbs)
        check("setup --dry-run prints the command without running it",
              rc == 0 and "verify.py" in out)

        # every get flag, together, on a real folder
        dbsf = os.path.join(tmp, "dbs-flags")
        os.makedirs(dbsf)
        rc, out = cli(["get", data, "--compact", "--dry-run"], dbsf)
        check("get --compact --dry-run plans compaction too",
              rc == 0 and "compact" in out.lower())
        rc, out = cli(["get", data, "--dest", os.path.join(tmp, "d1"),
                       "--filter", "nothing", "--dry-run"], dbsf)
        check("--dest and --filter are accepted for a local folder",
              rc == 0 and "Traceback" not in out)

        # ---- 1b. a broken environment must say so, not traceback ----
        # Every one of these used to die on a raw ModuleNotFoundError from
        # whichever module imported duckdb first, or worse, quietly succeed.
        bare = os.path.join(tmp, "bare-venv")
        rc = subprocess.run([sys.executable, "-m", "venv", bare],
                            capture_output=True).returncode
        if rc == 0 and os.path.exists(os.path.join(bare, "bin", "python")):
            nodeps = os.path.join(bare, "bin", "python")
            bad = []
            for cmd in (["status"], ["scan", empty], ["get", data],
                        ["serve", "--dry-run"], ["setup"], []):
                p = subprocess.run([nodeps, ATLAS] + cmd, env=env_for(dbs),
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, timeout=300)
                out = p.stdout + p.stderr
                name = " ".join(cmd) or "(no arguments)"
                if "Traceback" in out:
                    bad.append(f"{name}: traceback")
                elif p.returncode == 0:
                    bad.append(f"{name}: exited 0 with no dependencies")
                elif "Missing dependencies" not in out:
                    bad.append(f"{name}: {out.strip().splitlines()[-1][:60]}")
            check("with no dependencies every command says so and exits non-zero",
                  not bad, "; ".join(bad))
        else:
            print("  [ -- ] could not build a bare venv here, skipped")

        # A database directory that cannot be used must be named before any
        # ingest runs, not produce one DuckDB traceback per ingest script.
        for where, why in ((os.path.join(tmp, "nope"), "does not exist"),
                           (os.path.join(tmp, "nope", "deeper"),
                            "parent does not exist")):
            p = subprocess.run([sys.executable, ATLAS, "get", data],
                               env={**os.environ, "ATLAS_DB_DIR": where},
                               cwd=ROOT, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=300)
            out = p.stdout + p.stderr
            check(f"a database directory that {why} is reported, not crashed on",
                  p.returncode != 0 and "Traceback" not in out
                  and "database directory" in out,
                  out.strip().splitlines()[-1][:80] if out.strip() else "")

        # ATLAS_DB_DIR on its own has to actually move the databases: it is
        # documented to, and silently writing into the repository instead is
        # how someone loses track of where their data went.
        moved = os.path.join(tmp, "moved")
        os.makedirs(moved)
        p = subprocess.run([sys.executable, ATLAS, "get",
                            os.path.join(data, "mds2-test")],
                           env={**os.environ, "ATLAS_DB_DIR": moved}, cwd=ROOT,
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=1800)
        here = sorted(f for f in os.listdir(moved) if f.endswith(".duckdb"))
        check("ATLAS_DB_DIR alone puts the databases where it says",
              p.returncode == 0 and here and "ATLAS_DB_DIR" in (p.stdout + p.stderr),
              str(here))

        # `atlas.py status | head` closes the pipe mid-write.
        p = subprocess.run(f"{sys.executable} {ATLAS} status | head -2",
                           shell=True, env=env_for(moved), cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        check("a closed pipe (status | head) is not an error",
              p.returncode == 0 and "BrokenPipeError" not in p.stderr,
              p.stderr.strip().splitlines()[-1][:70] if p.stderr.strip() else "")

        # --ask with stdin closed outright raises RuntimeError, not EOFError.
        askdbs = os.path.join(tmp, "askdbs")
        os.makedirs(askdbs, exist_ok=True)
        for how, shell in (("closed", "0<&-"), ("redirected", "</dev/null")):
            p = subprocess.run(
                f"{sys.executable} {ATLAS} get {data} --ask {shell}",
                shell=True, env=env_for(askdbs), cwd=ROOT,
                capture_output=True, text=True, timeout=600)
            out = p.stdout + p.stderr
            check(f"--ask with stdin {how} refuses instead of crashing",
                  p.returncode != 0 and "Traceback" not in out
                  and "no terminal" in out,
                  out.strip().splitlines()[-1][:70] if out.strip() else "")

        # A database that is not usable must never be reported as healthy, and
        # nothing may be written into it. Every shape a bad file comes in.
        import duckdb as _dd
        cases = {}
        for label in ("zero", "text", "dir", "wrongcols", "foreign"):
            d = os.path.join(tmp, "bad-" + label)
            os.makedirs(d)
            iq = os.path.join(d, "iq.duckdb")
            if label == "zero":
                open(iq, "wb").close()
            elif label == "text":
                open(iq, "w").write("not a database at all")
            elif label == "dir":
                os.makedirs(iq)
            else:
                c2 = _dd.connect(iq)
                c2.execute("CREATE TABLE iq_stft (wrong_col INT, another VARCHAR)"
                           if label == "wrongcols"
                           else "CREATE TABLE customers (id INT, name VARCHAR)")
                c2.close()
            cases[label] = d
        bad = []
        for label, d in cases.items():
            e = {**os.environ, "ATLAS_DB_DIR": d}
            for cmd in (["status"], []):
                p = subprocess.run([sys.executable, ATLAS] + cmd, env=e,
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, timeout=300)
                out = p.stdout + p.stderr
                name = f"{label}/{' '.join(cmd) or 'menu'}"
                if "Traceback" in out:
                    bad.append(f"{name}: traceback")
                # The exact failure mode this guards: claiming a broken file is
                # a working pyramid and telling the user to start the viewer.
                elif "stft pyramid" in out or "Ready. Start the viewer" in out:
                    bad.append(f"{name}: reported as healthy")
        check(f"a broken iq.duckdb is never called healthy ({len(cases)} shapes)",
              not bad, "; ".join(bad))

        bad = []
        for label, d in cases.items():
            p = subprocess.run([sys.executable, ATLAS, "doctor"],
                               env={**os.environ, "ATLAS_DB_DIR": d}, cwd=ROOT,
                               stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=600)
            after = os.path.join(d, "iq.duckdb")
            if label == "dir":
                continue
            if label in ("wrongcols", "foreign"):
                # doctor used to build the demo straight into these, adding
                # tables to someone else's database with no backup.
                try:
                    c2 = _dd.connect(after, read_only=True)
                    tables = {r[0] for r in c2.execute(
                        "SELECT table_name FROM information_schema.tables"
                    ).fetchall()}
                    c2.close()
                except Exception:                             # noqa: BLE001
                    tables = set()
                if "iq_meta" in tables:
                    bad.append(f"{label}: doctor wrote into it ({sorted(tables)})")
            if "Traceback" in (p.stdout + p.stderr):
                bad.append(f"{label}: traceback")
        check("doctor never writes into a database it called unusable",
              not bad, "; ".join(bad))

        # A search that stops early has to say so. Silently missing data and
        # then blaming the file naming sends people the wrong way entirely.
        deep = os.path.join(tmp, "deep", *[f"lvl{i}" for i in range(19)])
        TI.make_data(deep)
        root_deep = os.path.join(tmp, "deep")
        rc, out = cli(["scan", root_deep], dbs)
        check("a scan that hit the depth limit says so instead of blaming names",
              rc != 0 and "directory levels" in out
              and "--max-depth" in out and "named differently" not in out,
              next((l.strip() for l in out.splitlines()
                    if "directory levels" in l), ""))
        rc, out = cli(["scan", root_deep, "--max-depth", "40"], dbs)
        check("--max-depth finds data the default depth misses",
              rc == 0 and "CBRS PSD export(s)" in out,
              next((l.strip() for l in out.splitlines()
                    if "PSD export" in l), ""))

        # ---- 2. prompt(): every kind of thing a person can type ----
        import atlas                                          # noqa: E402
        import builtins
        cases = {"1": 0, "3": 2, "03": 2, " 2 ": 1,            # accepted
                 "q": None, "Q": None, "quit": None, "exit": None, "": None,
                 "   ": None}
        rejects = ["0", "4", "-1", "1.5", "abc", "1a", "!!", "999999"]
        bad = []
        for raw, want in cases.items():
            typed = [raw]
            builtins.input = lambda p="", t=typed: t.pop(0)
            got = atlas.prompt(3)
            if got != want:
                bad.append(f"{raw!r} -> {got!r}, wanted {want!r}")
        check(f"prompt accepts and quits on all {len(cases)} valid inputs",
              not bad, "; ".join(bad))

        # each reject must be re-asked, then a valid answer taken
        bad = []
        for raw in rejects:
            typed = [raw, "2"]
            builtins.input = lambda p="", t=typed: t.pop(0)
            got = atlas.prompt(3)
            if got != 1 or typed:
                bad.append(f"{raw!r} -> {got!r}")
        check(f"prompt re-asks on all {len(rejects)} invalid inputs",
              not bad, "; ".join(bad))

        def eof(p=""):
            raise EOFError

        builtins.input = eof
        check("prompt returns NO_INPUT at end of input",
              atlas.prompt(3) is atlas.NO_INPUT)

        def interrupt(p=""):
            raise KeyboardInterrupt

        builtins.input = interrupt
        check("Ctrl+C at the prompt quits instead of crashing",
              atlas.prompt(3) is None)

        # ---- 3. the menu, in every state, pressing every number ----
        states = {}

        dbs_a = os.path.join(tmp, "state-empty")
        os.makedirs(dbs_a)
        states["nothing built, no data"] = ("-", dbs_a)

        dbs_b = os.path.join(tmp, "state-data")
        os.makedirs(dbs_b)
        states["data on disk, nothing built"] = (data, dbs_b)

        dbs_c = os.path.join(tmp, "state-built")
        os.makedirs(dbs_c)
        rc, out = cli(["get", os.path.join(data, "mds2-test")], dbs_c)
        built_ok = rc == 0 and os.path.exists(os.path.join(dbs_c, "psd.duckdb"))
        check("a state with databases already built could be set up", built_ok,
              out.strip().splitlines()[-1] if not built_ok else "")
        states["already built"] = ("-", dbs_c)

        dbs_d = os.path.join(tmp, "state-stray")
        os.makedirs(dbs_d)
        shutil.copy2(os.path.join(dbs_c, "psd.duckdb"),
                     os.path.join(dbs_d, "spectrum_viewer.db"))
        states["an oddly-named database sitting there"] = ("-", dbs_d)

        for label, (root, dbs_s) in states.items():
            rc, out = menu(harness, root, ["q"], dbs_s)
            entries = items_of(out)
            ok = (rc == 0 and entries and "nothing done" in out
                  and "Traceback" not in out)
            check(f"menu renders and quits cleanly: {label}", ok,
                  f"{len(entries)} choice(s): " + " | ".join(entries))
            if not entries:
                continue
            # Press every number this menu offers, one run each.
            for i in range(1, len(entries) + 1):
                sub = os.path.join(tmp, f"press-{abs(hash(label))%9999}-{i}")
                os.makedirs(sub, exist_ok=True)
                # A fresh db dir per press, except where the state IS the dbs
                use = dbs_s if i else dbs_s
                rc, out = menu(harness, root, [str(i)], use, timeout=1200)
                label_i = entries[i - 1][:46]
                # scan finding nothing is a legitimate non-zero exit
                allow_nonzero = "Search this whole machine" in entries[i - 1]
                ok = ("Traceback" not in out
                      and (rc == 0 or allow_nonzero))
                check(f"  press {i} ({label_i}) in [{label}]", ok,
                      f"rc={rc} " + (out.strip().splitlines()[-1][:90]
                                     if out.strip() else ""))
            # A number past the end must be re-asked, not accepted.
            rc, out = menu(harness, root, [str(len(entries) + 1), "q"], dbs_s)
            check(f"menu rejects an out-of-range number: {label}",
                  rc == 0 and "not a choice" in out and "nothing done" in out)

        # ---- 4. the nested dataset menus, downloading for real ----
        srv, port = TF.serve_fixture()
        fixture = os.path.join(tmp, "datasets.json")
        with open(fixture, "w") as f:
            json.dump({
                "_comment": "local fixture standing in for datasets.json",
                "fixture-iq": {"record": "mds2-test",
                               "title": "Synthetic IQ record",
                               "filter": "1.4MHz/config_0",
                               "sample": 2, "flags": ["--allow-http"]},
                "fixture-waf": {"record": "mds2-waf",
                                "title": "Record behind bot protection",
                                "note": "403s any non-browser client",
                                "sample": 2, "flags": ["--allow-http"]},
            }, f)
        pdr = {"ATLAS_PDR_BASE": f"http://127.0.0.1:{port}/rmm/records/{{}}",
               "ATLAS_DATASETS": fixture}

        dbs_m = os.path.join(tmp, "menu-dl")
        os.makedirs(dbs_m)
        rc, out = menu(harness, "-", ["2", "q"], dbs_m, pdr)
        names = items_of(out)
        check("the NIST menu entry lists the datasets it knows",
              rc == 0 and "Which dataset?" in out
              and any("fixture-iq" in n for n in names),
              " | ".join(names[-3:]))

        # Every dataset x every size option it actually offers, for real. The
        # size menu is two entries or three depending on whether the dataset
        # names a default filter, so the count is read rather than assumed.
        for di, dname in ((1, "fixture-iq"), (2, "fixture-waf")):
            probe = os.path.join(tmp, f"probe-{dname}")
            os.makedirs(probe)
            rc, out = menu(harness, "-", ["2", str(di), "q"], probe,
                           {**pdr, "ATLAS_DOWNLOAD_DIR":
                            os.path.join(probe, "downloads")})
            offered = [i for i in items_of(out) if "smallest" in i
                       or "slice" in i or "Everything" in i]
            check(f"the size menu for {dname} offers its options",
                  rc == 0 and len(offered) >= 2, " | ".join(offered))
            for si in range(1, len(offered) + 1):
                d = os.path.join(tmp, f"dl-{dname}-{si}")
                os.makedirs(d)
                # Its own download folder, so each option really downloads
                # rather than finding the previous option's files complete.
                rc, out = menu(harness, "-", ["2", str(di), str(si)], d,
                               {**pdr, "ATLAS_DOWNLOAD_DIR":
                                os.path.join(d, "downloads")}, timeout=1800)
                dbf = os.path.join(d, "iq.duckdb")
                ok = (rc == 0 and os.path.exists(dbf) and "ingested" in out
                      and "Traceback" not in out)
                check(f"  menu download+ingest: {dname}, {offered[si-1][:40]}",
                      ok, next((l.strip() for l in out.splitlines()
                                if "ingested" in l or "FAILED" in l
                                or "failed" in l), out.strip()[-110:]))

        # backing out of each nested menu must change nothing
        for answers, where in ((["2", "q"], "the dataset list"),
                               (["2", "1", "q"], "the size list")):
            d = os.path.join(tmp, "back-" + str(len(answers)))
            os.makedirs(d)
            rc, out = menu(harness, "-", answers, d,
                           {**pdr, "ATLAS_DOWNLOAD_DIR":
                            os.path.join(d, "downloads")})
            check(f"quitting {where} downloads nothing",
                  rc == 0 and not [f for f in os.listdir(d)
                                   if f.endswith(".duckdb")]
                  and "Traceback" not in out)

        # and the same dataset by name on the command line
        d = os.path.join(tmp, "byname")
        os.makedirs(d)
        rc, out = cli(["get", "fixture-iq", "--dest", os.path.join(d, "f")],
                      d, pdr, timeout=1800)
        check("a dataset name works on the command line, flags and all",
              rc == 0 and os.path.exists(os.path.join(d, "iq.duckdb")),
              next((l.strip() for l in out.splitlines()
                    if "ingested" in l or "failed" in l), out.strip()[-110:]))
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - every command, flag and menu choice behaves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
