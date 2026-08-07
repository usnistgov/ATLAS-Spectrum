"""test_atlas.py: prove `python atlas.py get <thing>` works it out for you.

The point of atlas.py is that a user should not have to know which of five
ingest scripts matches the bytes they have. This builds one folder containing
IQ captures, CBRS PSD exports, CBRS PFP exports and Summaries CSVs all mixed
together, hands the folder to `atlas.py get`, and checks that every layer ends
up built without naming a single ingest script.

Also covers: status on a fresh clone, --dry-run changing nothing, prebuilt
.duckdb files being installed instead of ingested, a folder holding nothing
recognisable, and friendly names from datasets.json.

    python examples/test_atlas.py          # exits 0 = PASS
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

import test_ingest as TI                                     # noqa: E402

failed = 0


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def make_iq(folder, n=1024 * 64):
    """A tiny SigMF capture. Default size is enough for one STFT level."""
    import numpy as np
    os.makedirs(folder, exist_ok=True)
    t = np.arange(n) / 2e6
    x = (np.exp(2j * np.pi * 3e5 * t) * 0.5).astype(np.complex64)
    x.tofile(os.path.join(folder, "demo.sigmf-data"))
    with open(os.path.join(folder, "demo.sigmf-meta"), "w") as f:
        json.dump({"global": {"core:datatype": "cf32_le",
                              "core:sample_rate": 2e6, "core:version": "1.0.0"},
                   "captures": [{"core:sample_start": 0,
                                 "core:frequency": 2.412e9}],
                   "annotations": []}, f)


def make_iqtar(folder, n=1024 * 64, fs=6.25e8, fc=9e8, truncate=False):
    """A tiny Rohde & Schwarz iq-tar capture, laid out as NIST mds2-3684 ships
    it: a `<name>.iq/` directory holding interleaved complex float32 beside the
    .xml that carries the sample rate and centre frequency."""
    import numpy as np
    d = os.path.join(folder, "1_synthetic_900_IQ_time.iq")
    os.makedirs(d, exist_ok=True)
    data = os.path.join(d, "File_20230227164018.complex1ch.float32")
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * 4e7 * t) * 0.5
    inter = np.empty(2 * n, dtype="<f4")
    inter[0::2], inter[1::2] = x.real, x.imag
    inter[:len(inter) // 2 if truncate else len(inter)].tofile(data)
    with open(data + ".xml", "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n<RS_IQ_TAR_FileFormat>'
            '<Name>FSW-8</Name><DateTime>2023-02-27 16:40:03</DateTime>'
            f'<Samples>{n}</Samples>'
            f'<Clock unit="Hz">{fs:.6f}</Clock>'
            '<Format>complex</Format><DataType>float32</DataType>'
            '<ScalingFactor unit="V">1</ScalingFactor>'
            '<NumberOfChannels>1</NumberOfChannels>'
            f'<DataFilename>{os.path.basename(data)}</DataFilename>'
            '<UserData><RohdeSchwarz><SpectrumAnalyzer>'
            '<Key name="Ch1_MeasBandwidth[Hz]">1.6e+08</Key>'
            '</SpectrumAnalyzer></RohdeSchwarz></UserData>'
            f'<CenterFrequency unit="Hz">{fc:.6f}</CenterFrequency>'
            '<PreviewData><ArrayOfFloat><float>-159</float>'
            '<float>-159</float></ArrayOfFloat></PreviewData>'
            '</RS_IQ_TAR_FileFormat>\n')
    return d, data


def run(args, dbs, extra=None):
    env = {**os.environ,
           "SPECTRUM_DB": os.path.join(dbs, "spectrum.duckdb"),
           "PSD_DB": os.path.join(dbs, "psd.duckdb"),
           "PFP_DB": os.path.join(dbs, "pfp.duckdb"),
           "IQ_DB": os.path.join(dbs, "iq.duckdb"),
           "ATLAS_DB_DIR": dbs, **(extra or {})}
    # stdin closed: --ask must not be able to block a non-interactive run
    p = subprocess.run([sys.executable, ATLAS] + args, env=env, cwd=ROOT,
                       stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main():
    print("atlas.py test\n")
    tmp = tempfile.mkdtemp(prefix="atlas-cli-test-")
    data = os.path.join(tmp, "everything")
    dbs = os.path.join(tmp, "dbs")
    os.makedirs(dbs)
    try:
        # one folder, four different kinds of data mixed together
        TI.make_data(data)
        make_iq(os.path.join(data, "mds2-test", "iq", "run1"))

        rc, out = run(["status"], dbs)
        check("status on a fresh clone says nothing is built",
              rc == 0 and "Nothing is built yet" in out and "absent" in out)

        rc, out = run(["get", data, "--dry-run"], dbs)
        check("--dry-run prints a plan naming every kind found",
              rc == 0 and "IQ captures in" in out and "CBRS PSD, sensor" in out
              and "CBRS PFP, sensor" in out and "CBRS Summaries in" in out,
              str(out.count("  - ")) + " planned step(s)")
        check("--dry-run builds nothing", not os.listdir(dbs))

        rc, out = run(["get", data], dbs)
        built = sorted(f for f in os.listdir(dbs) if f.endswith(".duckdb"))
        check("get on a mixed folder builds every layer, no script named",
              rc == 0 and built == ["iq.duckdb", "pfp.duckdb", "psd.duckdb",
                                    "spectrum.duckdb"], str(built))
        check("it finishes by reporting the new state",
              "Ready. Start the viewer with" in out)
        check("the sensor was discovered, not supplied",
              TI.SENSOR in out, TI.SENSOR)

        rc, out = run(["status"], dbs)
        check("status now shows each layer with its schema and size",
              rc == 0 and "uncompacted" in out and "stft pyramid" in out
              and "1 capture(s)" in out and "days" in out)

        # re-running must be safe: every ingest is resumable
        rc, out = run(["get", data], dbs)
        check("re-running get is idempotent",
              rc == 0 and ("already ingested" in out or "skip (already" in out))

        # prebuilt databases: install, do not ingest
        pre = os.path.join(tmp, "prebuilt")
        os.makedirs(pre)
        shutil.copy2(os.path.join(dbs, "psd.duckdb"),
                     os.path.join(pre, "psd.duckdb"))
        dbs2 = os.path.join(tmp, "dbs2")
        os.makedirs(dbs2)
        rc, out = run(["get", pre], dbs2)
        check("a folder of prebuilt .duckdb files skips ingest entirely",
              rc == 0 and "no ingest needed" in out
              and os.path.exists(os.path.join(dbs2, "psd.duckdb"))
              and "psd_ingest.py" not in out)

        rc, out = run(["get", pre], dbs2)
        check("re-loading the same prebuilt database adds nothing twice",
              rc == 0 and "nothing new" in out
              and not os.path.exists(os.path.join(dbs2, "psd.duckdb.bak")),
              next((l.strip() for l in out.splitlines()
                    if "nothing new" in l or "installed" in l), ""))

        # ---- databases you already have, loaded without losing any ----
        # Two IQ databases holding different captures. Copying the second over
        # the first is how one of them used to disappear silently.
        def iq_db(path, cid):
            c = _dd.connect(path)
            c.execute("""CREATE TABLE iq_meta (
                id VARCHAR PRIMARY KEY, dataset VARCHAR, name VARCHAR,
                path VARCHAR, fc DOUBLE, fs DOUBLE, duration DOUBLE,
                n_samples BIGINT, nfft INT, nfreq INT, hop INT, nlevels INT,
                qmin DOUBLE, qmax DOUBLE, vmin DOUBLE, vmax DOUBLE)""")
            c.execute("CREATE TABLE iq_stft (id VARCHAR, level INT, "
                      "col0 BIGINT, ncols INT, chunk BLOB)")
            c.execute("INSERT INTO iq_meta VALUES (?,'ds',?,'/p',2.4e9,2e6,1.0,"
                      "2048000,1024,1024,1024,1,-120,-20,-110,-30)", [cid, cid])
            c.execute("INSERT INTO iq_stft VALUES (?,0,0,4,?)",
                      [cid, bytes(4 * 1024)])
            c.close()

        import duckdb as _dd
        two = os.path.join(tmp, "twodbs")
        os.makedirs(two)
        iq_db(os.path.join(two, "alpha.duckdb"), "capture-A")
        iq_db(os.path.join(two, "beta.duckdb"), "capture-B")
        dbs9 = os.path.join(tmp, "dbs9")
        os.makedirs(dbs9)
        rc, out = run(["get", two], dbs9)

        def captures(path):
            c = _dd.connect(path, read_only=True)
            got = [r[0] for r in c.execute(
                "SELECT id FROM iq_meta ORDER BY 1").fetchall()]
            n = c.execute("SELECT count(*) FROM iq_stft").fetchone()[0]
            c.close()
            return got, n

        got, nstft = captures(os.path.join(dbs9, "iq.duckdb"))
        check("two databases of one kind are merged, not silently overwritten",
              rc == 0 and got == ["capture-A", "capture-B"] and nstft == 2,
              f"{got}, {nstft} stft row(s)")
        check("the plan says merge rather than claiming two installs",
              out.count("merge") >= 1 and "merged 1 new item(s)" in out,
              next((l.strip() for l in out.splitlines() if "merged" in l), ""))

        rc, out = run(["get", two], dbs9)
        got, nstft = captures(os.path.join(dbs9, "iq.duckdb"))
        check("merging is idempotent: a second run duplicates nothing",
              rc == 0 and got == ["capture-A", "capture-B"] and nstft == 2
              and "nothing new" in out, f"{got}, {nstft} stft row(s)")

        # A third database merges into the library that is already live.
        more = os.path.join(tmp, "onemore")
        os.makedirs(more)
        iq_db(os.path.join(more, "gamma.duckdb"), "capture-C")
        rc, out = run(["get", more], dbs9)
        got, _n = captures(os.path.join(dbs9, "iq.duckdb"))
        check("a later database adds to the live library instead of replacing it",
              rc == 0 and got == ["capture-A", "capture-B", "capture-C"],
              str(got))

        # doctor --adopt goes through the same path, so a stray database found
        # beside serve.py is merged in rather than replacing what is there.
        shutil.copy2(os.path.join(more, "gamma.duckdb"),
                     os.path.join(dbs9, "delta.duckdb"))
        iq_db(os.path.join(tmp, "eps.duckdb"), "capture-E")
        shutil.copy2(os.path.join(tmp, "eps.duckdb"),
                     os.path.join(dbs9, "eps.duckdb"))
        rc, out = run(["doctor", "--adopt"], dbs9)
        got, _n = captures(os.path.join(dbs9, "iq.duckdb"))
        check("doctor --adopt merges strays into the existing library",
              "capture-E" in got and "capture-A" in got,
              str(got))

        # spectrum cannot be merged (its pyramid is an aggregate), so a second
        # one must be refused out loud rather than replacing the first.
        def spec_db(path, sensor):
            c = _dd.connect(path)
            c.execute("CREATE TABLE raw (sensor VARCHAR, freq DOUBLE, "
                      "t DOUBLE, mx DOUBLE, md DOUBLE, mn DOUBLE)")
            c.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?)",
                          [(sensor, 3550.0, 1.7e9 + i, -70.0, -80.0, -85.0)
                           for i in range(4)])
            c.execute("CREATE TABLE meta AS SELECT sensor, min(t) AS t_min, "
                      "max(t) AS t_max, count(*) AS n FROM raw GROUP BY sensor")
            c.close()

        twospec = os.path.join(tmp, "twospec")
        os.makedirs(twospec)
        spec_db(os.path.join(twospec, "one.duckdb"), "S-ONE")
        spec_db(os.path.join(twospec, "two.duckdb"), "S-TWO")
        dbs10 = os.path.join(tmp, "dbs10")
        os.makedirs(dbs10)
        rc, out = run(["get", twospec], dbs10)
        check("an unmergeable second database is refused loudly, not applied",
              rc == 0 and "NOT loaded" in out and "cannot be merged" in out
              and "atlas.py get" in out,
              next((l.strip() for l in out.splitlines() if "NOT loaded" in l), ""))

        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        rc, out = run(["get", empty], dbs)
        check("a folder with nothing recognisable fails and says what it wanted",
              rc != 0 and "Nothing recognisable" in out
              and "Traceback" not in out)

        # ---- Rohde & Schwarz iq-tar, the format mds2-3684 ships ----
        sys.path.insert(0, os.path.join(ROOT, "ingest"))
        import sigmf_io                                       # noqa: E402
        iqt = os.path.join(tmp, "iqtar")
        _iqdir, iqfile = make_iqtar(iqt)
        cap = sigmf_io.open_capture(iqfile)
        check("an iq-tar capture reads its rate and centre from the .xml",
              (cap.sample_rate, cap.center_freq) == (6.25e8, 9e8)
              and cap.n_samples == 1024 * 64,
              f"fs={cap.sample_rate} fc={cap.center_freq} n={cap.n_samples}")
        check("the .xml can be opened instead of the samples file",
              sigmf_io.open_capture(iqfile + ".xml").n_samples == cap.n_samples)
        x = cap.read(0, 256)
        check("its samples come back as complex64 of the right length",
              x.dtype == __import__("numpy").complex64 and len(x) == 256,
              f"{x.dtype} len={len(x)}")

        # A part-downloaded capture must be named as such, not ingested short:
        # every axis is derived from the file's length, so a truncated one looks
        # perfectly healthy and simply shows less signal than it should.
        tr = os.path.join(tmp, "iqtar-short")
        _, short = make_iqtar(tr, truncate=True)
        try:
            sigmf_io.open_capture(short)
            ok, why = False, "a half-length capture was accepted"
        except ValueError as e:
            ok, why = "truncated" in str(e), str(e)[:110]
        check("a truncated iq-tar capture is refused, not silently shortened",
              ok, why)

        # And without the sidecar there is no sample rate to be had, so say so
        # rather than inventing one.
        os.remove(iqfile + ".xml")
        try:
            sigmf_io.open_capture(iqfile)
            ok, why = False, "a capture with no .xml was accepted"
        except FileNotFoundError as e:
            ok, why = ".xml" in str(e), str(e)[:110]
        check("an iq-tar capture with no .xml names what is missing", ok, why)

        make_iqtar(iqt)     # restore the sidecar, then ingest for real
        rc, out = run(["get", iqt], dbs)
        check("atlas.py get ingests an iq-tar folder without being told how",
              rc == 0 and "1 ingested" in out and "Traceback" not in out,
              next((l.strip() for l in out.splitlines()
                    if "ingested" in l), out[-160:]))

        # friendly names resolve without touching the network (dry run)
        names = json.load(open(os.path.join(ROOT, "datasets.json")))
        real = [k for k in names if not k.startswith("_")]
        check("datasets.json parses and has entries", bool(real), str(real))
        rc, out = run(["get", "lte-uplink", "--dry-run"], dbs)
        check("a friendly name resolves to its record and default filter",
              rc == 0 and "mds2-3177" in out and "1.4MHz" in out
              and "fetch.py" in out,
              next((l.strip() for l in out.splitlines() if "fetch.py" in l), ""))

        # Every entry must name a record and say which reader handles it, so a
        # record that downloads but cannot be ingested is known before the
        # transfer rather than after it.
        bad = [k for k in real
               if not names[k].get("record") or not names[k].get("reads")]
        check("every dataset entry names a record and its reader", not bad,
              str(bad))

        rc, out = run(["get", "no-such-name-here", "--dry-run"], dbs)
        check("an unknown name is passed through to fetch, not crashed on",
              rc == 0 and "no-such-name-here" in out and "Traceback" not in out)

        rc, out = run(["get", "lte-uplink", "--filter", "", "--dry-run"], dbs)
        check("an explicit empty --filter beats the dataset default",
              rc == 0 and "--filter" not in out.split("$ ")[-1],
              next((l.strip() for l in out.splitlines() if "fetch.py" in l), ""))

        rc, out = run(["get", "mds2-3177", "--dry-run", "--nist-only",
                       "--allow-host", "example.org"], dbs)
        check("unknown flags are forwarded to fetch.py",
              rc == 0 and "--nist-only" in out and "--allow-host example.org" in out,
              next((l.strip() for l in out.splitlines() if "fetch.py" in l), ""))

        # A download folder of its own, so the assertion below does not depend
        # on where downloads happen to go on the machine running the test.
        dl = os.path.join(tmp, "dlhome")
        rc, out = run(["get", "<https://data.nist.gov/od/id/mds2-3177>",
                       "--dry-run"], dbs, {"ATLAS_DOWNLOAD_DIR": dl})
        # normalise the separator: the folder name is what is being checked,
        # and Windows spells the same path with backslashes
        flat = out.replace("\\", "/")
        want = (dl + "/mds2-3177").replace("\\", "/")
        check("a URL pasted with angle brackets still names the folder cleanly",
              rc == 0 and f"--dest {want}" in flat and ">" not in out,
              next((l.strip() for l in out.splitlines() if "fetch.py" in l), ""))

        dbs5 = os.path.join(tmp, "dbs5")
        os.makedirs(dbs5)
        rc, out = run(["get", data, "--ask"], dbs5)
        check("--ask with no terminal refuses rather than running blind",
              rc != 0 and "no terminal" in out and not os.listdir(dbs5))

        rc, out = run(["status", "--not-a-flag"], dbs)
        check("a typo on another subcommand is still an error",
              rc != 0 and "unrecognized arguments" in out)

        # a metadata CSV must not be mistaken for a Summaries export
        junk = os.path.join(tmp, "junk")
        os.makedirs(junk)
        with open(os.path.join(junk, "file_manifest.csv"), "w") as f:
            f.write("filename,checksum\nx.tdms,abc\n")
        rc, out = run(["get", junk], dbs)
        check("a non-Summaries CSV is reported, not handed to build_db",
              rc != 0 and "none has a Summaries header" in out
              and "build_db.py" not in out, "")

        # ---- the no-argument path ----
        dbs8 = os.path.join(tmp, "dbs8")
        os.makedirs(dbs8)
        rc, out = run([], dbs8)
        check("bare 'python atlas.py' reports the situation without being asked",
              rc == 0 and "checking this machine" in out
              and "nothing built yet" in out and "disk" in out)
        check("with no terminal it recommends rather than guessing",
              "Recommended:" in out and "Choose 1" not in out)
        check("a no-terminal run changes nothing", not os.listdir(dbs8))

        menu = os.path.join(tmp, "menu.py")
        with open(menu, "w") as f:
            f.write(
                "import sys, builtins\n"
                f"sys.path.insert(0, {ROOT!r})\n"
                "import atlas\n"
                # interactive() checks stdout too, which is a pipe here
                "atlas.interactive = lambda: True\n"
                "builtins.input = lambda p='': sys.argv[2]\n"
                "atlas.default_roots = lambda: [sys.argv[1]]\n"
                "sys.exit(atlas.cmd_auto(atlas.ns()))\n")

        env8 = {"SPECTRUM_DB": os.path.join(dbs8, "spectrum.duckdb"),
                "PSD_DB": os.path.join(dbs8, "psd.duckdb"),
                "PFP_DB": os.path.join(dbs8, "pfp.duckdb"),
                "IQ_DB": os.path.join(dbs8, "iq.duckdb"),
                "ATLAS_DB_DIR": dbs8}
        p = subprocess.run([sys.executable, menu, data, "q"],
                           env={**os.environ, **env8}, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        mo = p.stdout + p.stderr
        check("with a terminal it offers a numbered menu",
              "What would you like to do?" in mo and "1)" in mo
              and "<- recommended" in mo)
        check("the recommendation is to load the data it found",
              "1) Load the data in" in mo,
              next((l.strip() for l in mo.splitlines() if l.strip().startswith("1)")), ""))
        check("quitting the menu does nothing",
              "nothing done" in mo and not os.listdir(dbs8))

        # Windows reports NUL as a terminal, so a redirected run can look
        # interactive and then hit EOF on the first read. Both layers of the
        # guard are checked: the menu must be skipped, and if it somehow is
        # not, EOF must fall back to advice rather than "nothing done".
        eof_probe = os.path.join(tmp, "eof.py")
        with open(eof_probe, "w") as f:
            f.write(
                "import sys, builtins\n"
                f"sys.path.insert(0, {ROOT!r})\n"
                "import atlas\n"
                "atlas.interactive = lambda: True\n"
                "def eof(p=''):\n"
                "    print(p)\n"
                "    raise EOFError\n"
                "builtins.input = eof\n"
                "atlas.default_roots = lambda: []\n"
                "sys.exit(atlas.cmd_auto(atlas.ns()))\n")
        p = subprocess.run([sys.executable, eof_probe],
                           env={**os.environ, **env8}, cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        eo = p.stdout + p.stderr
        check("a terminal that cannot actually be read falls back to advice",
              p.returncode == 0 and "Recommended:" in eo
              and "nothing done" not in eo,
              next((l.strip() for l in eo.splitlines()
                    if l.startswith("Recommended")), ""))

        p = subprocess.run([sys.executable, menu, data, "1"],
                           env={**os.environ, **env8}, cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        built8 = sorted(f for f in os.listdir(dbs8) if f.endswith(".duckdb"))
        check("choosing 1 runs the whole ingest",
              p.returncode == 0 and built8 == ["iq.duckdb", "pfp.duckdb",
                                               "psd.duckdb", "spectrum.duckdb"],
              str(built8))

        # ---- doctor: one command on a hostile machine ----
        import duckdb
        legacy = os.path.join(tmp, "legacyrepo")
        os.makedirs(legacy)
        # a real PSD database under a name nothing would match on
        lp = os.path.join(legacy, "spectrum_viewer.db")
        con = duckdb.connect(lp)
        con.execute("CREATE TABLE psd (sensor VARCHAR, t DOUBLE, spec BLOB)")
        con.executemany("INSERT INTO psd VALUES (?,?,?)",
                        [("LEGACY", 1.7e9 + i * 240, bytes(2250)) for i in range(8)])
        con.execute("""CREATE TABLE psd_meta (sensor VARCHAR, f0 DOUBLE, df DOUBLE,
            nf INT, qmin DOUBLE, qmax DOUBLE, t_min DOUBLE, t_max DOUBLE,
            captures BIGINT)""")
        con.execute("INSERT INTO psd_meta VALUES ('LEGACY',3530040000.0,80000.0,"
                    "2250,-180.0,-90.0,1.7e9,1700001920.0,8)")
        con.close()
        with open(os.path.join(legacy, "notes.db"), "w") as f:
            f.write("not a database")

        import atlas                                        # noqa: E402
        name, label, rows = atlas.identify_db(lp)
        check("a database is identified by its tables, not its filename",
              (name, rows) == ("psd", 8), f"{name} / {label} / {rows} rows")
        name2, why, _ = atlas.identify_db(os.path.join(legacy, "notes.db"))
        check("a file that only looks like a database is rejected with a reason",
              name2 is None and "not readable" in why, why[:50])

        dbs6 = os.path.join(tmp, "dbs6")
        os.makedirs(dbs6)
        shutil.copy2(lp, os.path.join(dbs6, "spectrum_viewer.db"))
        shutil.copy2(os.path.join(legacy, "notes.db"),
                     os.path.join(dbs6, "notes.db"))

        rc, out = run(["doctor", data], dbs6)
        check("doctor spots an oddly-named database and says what it holds",
              "spectrum_viewer.db holds psd data" in out
              and "re-run with --adopt" in out,
              next((l.strip() for l in out.splitlines() if "holds psd" in l), ""))
        check("doctor ignores a file that only looks like a database",
              "notes.db ignored" in out)
        check("an unadopted database is left where it was",
              not os.path.exists(os.path.join(dbs6, "psd.duckdb")))

        rc2, out2 = run(["doctor", "--adopt", data], dbs6)
        check("doctor --adopt installs it where serve.py looks",
              "adopted spectrum_viewer.db as psd.duckdb" in out2
              and os.path.exists(os.path.join(dbs6, "psd.duckdb")))
        real, _lbl, nrows = atlas.identify_db(os.path.join(dbs6, "psd.duckdb"))
        check("the adopted database really is the PSD data",
              (real, nrows) == ("psd", 8), f"{real} / {nrows} rows")
        check("doctor checks environment, deps, disk, databases and data",
              rc == 0 and "1. Python and environment" in out
              and "2. Dependencies" in out and "3. Disk space" in out
              and "4. Databases" in out and "5. Spectrum data" in out
              and "7. End-to-end verification" in out)
        check("doctor builds the demo and verifies it end to end",
              "RESULT: PASS" in out and "VERDICT: working" in out)
        check("doctor finds source data and prints the command to load it",
              f'python atlas.py get "{data}"' in out)
        check("doctor reports the free space it checked",
              "free" in out and "Disk space" in out)

        rc, out = run(["doctor", "--dry-run"], dbs6)
        check("doctor --dry-run changes nothing",
              rc == 0 and "nothing will change" in out)

        # ---- scan: find data without being told where it is ----
        before_scan = sorted(os.listdir(dbs))
        junkdir = os.path.join(tmp, "scanjunk")
        os.makedirs(junkdir)
        with open(os.path.join(junkdir, "notes.txt"), "w") as f:
            f.write("hi")
        # a CBRS export named the wrong way round: the realistic miss case
        with open(os.path.join(junkdir, "SENSOR_2024-01-01_max.csv"), "w") as f:
            f.write("a,b\n1,2\n")

        rc, out = run(["scan", data, junkdir], dbs)
        check("scan finds data buried in a folder layout it was never told",
              rc == 0 and "CBRS PSD export(s)" in out
              and "CBRS PFP export(s)" in out and "IQ capture file(s)" in out,
              next((l.strip() for l in out.splitlines() if "PSD export" in l), ""))
        check("scan names the sensors it found",
              TI.SENSOR in out, TI.SENSOR)
        check("scan reports files it did not recognise",
              "not recognised" in out and ".txt" in out)
        check("scan warns loudly about unmatched CSVs and shows the patterns",
              "were found but not recognised" in out
              and "<YYYY-MM-DD>_<sensor>_<stat>.csv" in out)
        check("scan ends with the exact command to run",
              f'python atlas.py get "{data}"' in out)
        check("scan does not count a .sigmf-data companion as unrecognised",
              ".sigmf-data" not in out)

        rc, out = run(["scan", os.path.join(tmp, "does-not-exist")], dbs)
        check("scan says when a folder does not exist",
              rc != 0 and "does not exist" in out)

        rc, out = run(["scan", junkdir], dbs)
        check("scan exits non-zero when it finds nothing usable",
              rc != 0 and "Found nothing to ingest" in out)

        check("scan changed nothing", sorted(os.listdir(dbs)) == before_scan)

        # a single FILE means that file, not everything sitting beside it
        pair = os.path.join(tmp, "pair")
        make_iq(pair)
        for ext in ("sigmf-meta", "sigmf-data"):
            shutil.copy2(os.path.join(pair, f"demo.{ext}"),
                         os.path.join(pair, f"sibling.{ext}"))
        dbs3 = os.path.join(tmp, "dbs3")
        os.makedirs(dbs3)
        rc, out = run(["get", os.path.join(pair, "demo.sigmf-meta")], dbs3)
        check("one capture file ingests one capture, not its siblings",
              rc == 0 and "1 capture file(s)" in out and "1 ingested" in out
              and "--dataset pair" in out,
              next((l.strip() for l in out.splitlines() if "ingested" in l), ""))

        one_csv = os.path.join(data, "mds2-test", "sea", "spectra", "level-1",
                               f"{TI.DAYS[0]}_{TI.SENSOR}_max.csv")
        dbs4 = os.path.join(tmp, "dbs4")
        os.makedirs(dbs4)
        rc, out = run(["get", one_csv], dbs4)
        check("a single CBRS CSV is recognised, not rejected as unknown",
              rc == 0 and "CBRS PSD, sensor" in out
              and os.path.exists(os.path.join(dbs4, "psd.duckdb")),
              next((l.strip() for l in out.splitlines() if "  - " in l), ""))

        # a capture too short for an STFT must say so, and the failure must
        # reach the caller instead of looking like a clean run
        tiny = os.path.join(tmp, "tinyiq")
        make_iq(tiny, n=512)
        rc, out = run(["get", tiny], dbs)
        check("a capture too short for an STFT explains itself and fails",
              rc != 0 and "too short" in out and "at least 1024" in out
              and "Nothing was ingested" in out,
              next((l.strip() for l in out.splitlines() if "ERR " in l), ""))

        # a renamed ATLAS database IS recognised, by its tables
        stray = os.path.join(tmp, "stray")
        os.makedirs(stray)
        shutil.copy2(os.path.join(dbs, "psd.duckdb"),
                     os.path.join(stray, "my_capture.duckdb"))
        dbs7 = os.path.join(tmp, "dbs7")
        os.makedirs(dbs7)
        rc, out = run(["get", stray], dbs7)
        check("a renamed ATLAS database is recognised by its tables",
              rc == 0 and "no ingest needed" in out
              and os.path.exists(os.path.join(dbs7, "psd.duckdb")))

        # a database that is not ours must NOT be installed
        import duckdb as _dd
        foreign = os.path.join(tmp, "foreign")
        os.makedirs(foreign)
        fc = _dd.connect(os.path.join(foreign, "psd.duckdb"))
        fc.execute("CREATE TABLE customers (id INT, name VARCHAR)")
        fc.close()
        rc, out = run(["get", foreign], dbs7)
        # Refused, and said which file and why -- "nothing recognisable here" is
        # misleading when the thing it will not load is sitting right there.
        check("a foreign database is refused, and named, when it looks like ours",
              rc != 0 and "could not be read" in out
              and "psd.duckdb" in out and "customers" in out
              and "Traceback" not in out,
              next((l.strip() for l in out.splitlines()
                    if "customers" in l), out.strip()[-120:]))

        # ---- download a record and ingest it, in one command ----
        # Everything above hands `get` bytes that were already on disk. This is
        # the path that was broken: resolve a record, download it, work out
        # which ingest it needs, run it, and end with something renderable.
        # Both records are served by the fixture in test_fetch.py; mds2-waf
        # additionally 403s any client that does not look like a browser, which
        # is what data.nist.gov does and what used to stop this dead.
        import test_fetch as TF                              # noqa: E402
        srv, port = TF.serve_fixture()
        try:
            pdr = {"ATLAS_PDR_BASE": f"http://127.0.0.1:{port}/rmm/records/{{}}"}
            for rid, what in (("mds2-test", "a record"),
                              ("mds2-waf", "a record behind bot protection")):
                d = os.path.join(tmp, "dl-" + rid)
                os.makedirs(d)
                dl = os.path.join(d, "files")
                rc, out = run(["get", rid, "--dest", dl, "--filter",
                               "1.4MHz/config_0", "--allow-http"], d, pdr)
                iq = os.path.join(d, "iq.duckdb")
                check(f"get downloads and ingests {what} in one command",
                      rc == 0 and os.path.exists(iq)
                      and "2 downloaded" in out and "1 ingested" in out,
                      next((l.strip() for l in out.splitlines()
                            if "ingested" in l or "FAILED" in l), out[-200:]))
                check(f"the downloaded files really are on disk for {rid}",
                      os.path.exists(os.path.join(
                          dl, "1.4MHz", "config_0", "capture.sigmf-data")))
                # The database has to be usable, not merely present: verify.py
                # renders a tile out of it through the real Flask app.
                v = subprocess.run(
                    [sys.executable, os.path.join(HERE, "verify.py")],
                    env={**os.environ, "IQ_DB": iq, "ATLAS_DB_DIR": d,
                         "SPECTRUM_DB": os.path.join(d, "spectrum.duckdb"),
                         "PSD_DB": os.path.join(d, "psd.duckdb"),
                         "PFP_DB": os.path.join(d, "pfp.duckdb")},
                    cwd=ROOT, capture_output=True, text=True, timeout=600)
                vo = v.stdout + v.stderr
                check(f"the capture downloaded from {rid} renders a tile",
                      v.returncode == 0 and "RESULT: PASS" in vo
                      and "renders a spectrogram tile" in vo,
                      next((l.strip() for l in vo.splitlines()
                            if "iq_layer" in l), vo.strip()[-160:]))

            # A record that cannot be resolved must fail loudly and say what to
            # do, and must not leave a half-built database behind.
            d = os.path.join(tmp, "dl-blocked")
            os.makedirs(d)
            rc, out = run(["get", "mds2-waf", "--dest", os.path.join(d, "f"),
                           "--allow-http"], d,
                          {**pdr, "ATLAS_USER_AGENT": "some-scraper/1.0"})
            check("a download that is refused reports it and ingests nothing",
                  rc != 0 and "nothing was ingested" in out
                  and "python atlas.py get" in out and "Traceback" not in out
                  and not [f for f in os.listdir(d) if f.endswith(".duckdb")],
                  next((l.strip() for l in out.splitlines()
                        if "403" in l), out.strip()[-160:]))
        finally:
            srv.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - one command handles data you have and data you fetch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
