#!/usr/bin/env python3
"""
atlas.py: one command for the whole pipeline.

    python atlas.py

That is it. With no arguments it checks the machine, works out which situation
you are in, and offers the fix as a numbered choice. Everything below is the
same thing with the choice made for you, for scripts and for people who
already know what they want.

    python atlas.py doctor            check everything, fix what is safe
    python atlas.py scan              find spectrum data on this machine
    python atlas.py status            what is built, what to run next
    python atlas.py get <thing>       fetch if needed, ingest, done
    python atlas.py serve             start the viewer

<thing> can be any of:

    a folder or file already on disk   ./SEA-DATA, ~/Downloads/mds2-3177, capture.tdms
    a friendly name                    lte-uplink          (see datasets.json)
    a PDR record id or ark             mds2-3177, ark:/88434/mds2-3177
    a DOI or landing URL               https://doi.org/10.18434/mds2-3177
    any direct file URL                https://example.org/capture.sigmf-data

For a folder, every file in it is classified and the matching ingest runs:
SigMF/TDMS/npy captures, CBRS PSD exports, CBRS PFP exports, Summaries CSVs,
and prebuilt .duckdb databases are each recognised on sight. A record that
publishes finished .duckdb files is simply copied into place with no ingest at
all, which is the fastest path if one exists for your dataset.

The individual scripts under ingest/ all still work on their own; this drives
them, it does not replace them.
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ING = os.path.join(HERE, "ingest")
sys.path.insert(0, ING)

# Friendly dataset names. Overridable so you can keep your own list of records
# outside the repo (and so the tests can point the menu at a local fixture).
DATASETS = os.environ.get("ATLAS_DATASETS", os.path.join(HERE, "datasets.json"))
DBS = ("spectrum", "psd", "pfp", "iq")
IQ_EXT = (".sigmf-meta", ".tdms", ".npy", ".complex1ch.float32")
# Files that belong to a capture but are not the thing you point an
# ingest at. Counting these as "unrecognised" would be misleading.
COMPANION_EXT = (".sigmf-data", ".sha256")

# Directories a search never needs to descend into. Skipping these is the
# difference between scanning a home folder in seconds and in hours.
SKIP_DIRS = {
    "__pycache__", "node_modules", "site-packages", "venv", "env",
    "AppData", "Library", "Windows", "Program Files", "Program Files (x86)",
    "ProgramData", "$Recycle.Bin", "System Volume Information", "OneDriveTemp",
    "Application Data", "Local Settings", "Recovery", "PerfLogs", "tmp",
}

# Which table proves a database actually holds data, per schema.
SCHEMAS = {
    "spectrum": [("raw", "raw")],
    "psd": [("psd_chunk", "compacted"), ("psd", "uncompacted")],
    "pfp": [("pfp_chunk", "compacted"), ("pfp", "uncompacted")],
    "iq": [("iq_stft", "stft pyramid")],
}

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


DB_ENV = {"spectrum": "SPECTRUM_DB", "psd": "PSD_DB",
          "pfp": "PFP_DB", "iq": "IQ_DB"}


def db_path(name):
    """Where one database lives: its own variable first, then ATLAS_DB_DIR."""
    return os.environ.get(DB_ENV[name]) or os.path.join(
        db_dir(), f"{name}.duckdb")


def download_dir():
    """Where `get` puts a download when no --dest is given.

    Overridable because these records run to hundreds of gigabytes and the
    repository is often on the smallest disk in the machine.
    """
    return os.environ.get("ATLAS_DOWNLOAD_DIR", os.path.join(HERE, "downloads"))


def db_dir():
    """Where the databases live. Overridable so a test, or a second dataset,
    can keep its files somewhere other than beside serve.py.

    serve.py and every ingest script read ATLAS_DB_DIR the same way, so setting
    it is enough on its own; the per-database variables override it one at a
    time when only one file needs to move.
    """
    return os.path.abspath(os.environ.get("ATLAS_DB_DIR") or HERE)


def db_dir_problem():
    """None if the database directory can be written to, else why not.

    Checked before anything is built, because the alternative is one DuckDB
    IOException traceback per ingest script and no statement of the cause.
    """
    where = db_dir()
    if not os.path.isdir(where):
        parent = os.path.dirname(where.rstrip(os.sep))
        why = ("its parent does not exist either"
               if parent and not os.path.isdir(parent) else "it does not exist")
        return (f"the database directory {where} cannot be used: {why}.\n"
                "  Create it, or unset ATLAS_DB_DIR to use the repository "
                "folder.")
    if not os.access(where, os.W_OK):
        return (f"the database directory {where} is not writable "
                "(read-only filesystem, or owned by another user).\n"
                "  Point ATLAS_DB_DIR somewhere you can write.")
    return None


def short(p):
    """Repo-relative when that is actually shorter, absolute otherwise."""
    try:
        rel = os.path.relpath(p, HERE)
    except ValueError:              # different drive on Windows
        return p
    return rel if not rel.startswith("..") else p


def last_line(text, width=140):
    """The last line of a captured child's output that says anything.

    A traceback's last line is usually the exception, which is the one line
    worth surfacing when the rest has been captured. Not always, though: a
    DuckDB CatalogException prints the offending statement and a bare "^" caret
    underneath its message, so taking the literal last line reported an entire
    failed verification as "  [FAIL] ...  ^". Walk back to the line that names
    an exception, and fall back to the last non-empty line only if none does.
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return "(no output)"
    for l in reversed(lines[-15:]):
        head = l.split(":", 1)[0]
        if ":" in l and " " not in head and head.endswith(
                ("Error", "Exception", "Exit", "Interrupt")):
            return l[:width]
    return lines[-1][:width]


def run(argv, dry=False):
    """Run one step, echoing it first so the transcript is reproducible."""
    shown = " ".join(("python" if a == sys.executable else
                      short(a) if a.startswith(HERE) else a)
                     for a in argv)
    print(f"\n$ {shown}", flush=True)
    if dry:
        return 0
    return subprocess.run(argv, cwd=HERE).returncode


def script(name, *args):
    return [sys.executable, os.path.join(ING, name), *[str(a) for a in args]]


def load_datasets():
    try:
        with open(DATASETS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ---- classify what is on disk -----------------------------------------

def looks_like_summaries(path):
    """A Summaries CSV names its columns. Metadata and readme CSVs do not, and
    handing those to build_db.py just produces a confusing failure."""
    try:
        with open(path, "r", errors="replace") as f:
            head = f.readline().lower()
    except OSError:
        return False
    return "sensor_name" in head and "channel_frequency_mhz" in head


_NAME_RES = None


def name_res():
    """(psd, pfp) filename patterns, loaded from the ingest scripts themselves
    so there is exactly one definition of what a CBRS export is called."""
    global _NAME_RES
    if _NAME_RES is None:
        import psd_ingest
        import pfp_ingest
        _NAME_RES = (psd_ingest.NAME_RE, pfp_ingest.NAME_RE)
    return _NAME_RES


def kind_of(dirpath, name):
    """What one file is: 'iq' | 'duckdb' | 'pfp' | 'psd' | 'summaries' | None.

    Everything is decided by extension, by filename shape, or by a CSV's own
    header. Nothing here depends on the folder a file happens to sit in.
    """
    psd_re, pfp_re = name_res()
    low = name.lower()
    if low.endswith(IQ_EXT):
        return "iq"
    if low.endswith(DB_EXT):
        return "duckdb"
    if pfp_re.match(name):
        return "pfp"
    if psd_re.match(name):
        return "psd"
    if low.endswith(".csv"):
        return "summaries" if looks_like_summaries(
            os.path.join(dirpath, name)) else None
    return None


def walk_limited(root, max_files=200000, max_depth=12):
    """Yield (dirpath, name) under root, skipping system and toolchain noise.

    Returns (generator, limits) where limits records which budget ran out:
    'files' for the file cap, 'depth' for the directory-depth cap. Both are
    reported, because a search that quietly stopped early and then said "found
    nothing" sent people looking for a naming problem they did not have.
    """
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    limits = {"files": False, "depth": 0}
    n = 0

    def gen():
        nonlocal n
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            if dirpath.count(os.sep) - base >= max_depth:
                if dirnames:
                    limits["depth"] += len(dirnames)
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                yield dirpath, name
                n += 1
                if n >= max_files:
                    limits["files"] = True
                    return
    return gen(), limits


def classify(root):
    """Walk root -> what kinds of data are in it, and where.

    Returns a dict of kind -> sorted list of directories (or files) to hand to
    the matching ingest script.
    """
    out = {"iq": set(), "psd": set(), "pfp": set(), "csv": set(),
           "duckdb": [], "other": []}
    # A single file means that file, not everything sitting beside it. The CBRS
    # ingests take a directory, so those get the file's parent; the IQ ingest
    # takes either, so it gets the file itself.
    one = os.path.isfile(root)
    if one:
        names = [(os.path.dirname(root) or ".", os.path.basename(root))]
    else:
        names = [(dp, n) for dp, _, ns in os.walk(root) for n in ns]
    for dirpath, n in names:
        kind = kind_of(dirpath, n)
        if kind == "iq":
            out["iq"].add(os.path.join(dirpath, n) if one else root)
        elif kind == "duckdb":
            out["duckdb"].append(os.path.join(dirpath, n))
        elif kind == "pfp":
            out["pfp"].add(dirpath if one else root)
        elif kind == "psd":
            out["psd"].add(dirpath if one else root)
        elif kind == "summaries":
            out["csv"].add(dirpath)
        elif n.lower().endswith(".csv"):
            out["other"].append(os.path.join(dirpath, n))
    out["duckdb"].sort()
    return out


def prebuilt_plan(found):
    """-> list of (src, dst, kind, action) for every prebuilt database found.

    'install' copies a file into place. 'merge' folds its data into whatever is
    already there, which is what happens whenever the destination already holds
    usable data of that kind, or a second database of the same kind turns up in
    the same folder. 'skip' is for the one case neither works: more than one
    database of a kind that cannot be merged.

    Also returns the database-shaped files that could NOT be read as one of
    ours, with the reason. They are reported rather than dropped, because
    "nothing recognisable here" is a lie when a corrupt .duckdb is sitting right
    there.
    """
    by_kind, out, rejected = {}, [], []
    for p in found["duckdb"]:
        # Identify by the tables inside, so spectrum_viewer.db or any other
        # name is recognised for what it holds.
        real, why, rows = identify_db(p)
        if real:
            by_kind.setdefault(real, []).append((p, rows))
        else:
            rejected.append((p, why))
    for kind, entries in sorted(by_kind.items()):
        # Biggest first, so the base is the one that needs the fewest inserts.
        entries.sort(key=lambda e: (-e[1], e[0]))
        dst = db_path(kind)
        # A file that IS the destination is not a candidate to install over
        # itself. `python atlas.py get .` from the repository found
        # spectrum.duckdb, moved it to spectrum.duckdb.bak, and then failed to
        # copy it back from the path it had just emptied -- a traceback that
        # left the viewer's database missing. Report those as already installed
        # instead, which is also the honest answer for re-running `get` on a
        # folder that was ingested before.
        already = [e for e in entries if same_file(e[0], dst)]
        entries = [e for e in entries if not same_file(e[0], dst)]
        for p, _rows in already:
            out.append((p, dst, kind, "already"))
        live = False
        if os.path.exists(dst):
            real, _label, rows = identify_db(dst)
            live = real == kind and rows > 0
        for i, (p, _rows) in enumerate(entries):
            if kind in MERGEABLE and (live or i > 0):
                out.append((p, dst, kind, "merge"))
            elif live or i > 0:
                # Installing over data that cannot be merged destroys it. This
                # used to test i > 0 alone, so `get <folder>` holding one
                # spectrum database replaced a live spectrum.duckdb -- a full
                # ingest of someone's own CSVs, gone but for the .bak, with the
                # plan line reading only "install prebuilt spectrum.duckdb".
                out.append((p, dst, kind, "skip"))
            else:
                out.append((p, dst, kind, "install"))
    return out, rejected


def same_file(a, b):
    """True when two paths name the same file on disk.

    os.path.samefile is the reliable test -- it follows symlinks and on Windows
    compares the volume's file id, so C:\\ATLAS\\x.duckdb and
    c:\\atlas\\..\\ATLAS\\x.duckdb are the same file -- but it raises when
    either path is missing. Fall back to a normalised string compare, which is
    also what Windows' case-insensitivity needs.
    """
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return (os.path.normcase(os.path.abspath(a))
                == os.path.normcase(os.path.abspath(b)))


def discard(*paths):
    """Delete half-finished copies. Best effort: they are only ever temporary."""
    for p in paths:
        try:
            if p:
                os.remove(p)
        except OSError:
            pass


def make_writable(path):
    """Clear the read-only bit that a copy inherits from its source.

    Databases handed over on a DVD, unpacked from an archive, or copied off a
    read-only share arrive read-only, and copy2 copies the mode along with the
    bytes. Every ingest then fails to open its own destination for writing, and
    on Windows the file cannot even be replaced until the bit is cleared.
    """
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)
    except OSError:
        pass


def install_db(src, dst):
    """Copy one database file into place. Returns a line describing what it did.

    Every failure mode of a file copy has to leave the existing database
    intact -- this is the one operation in atlas.py that can destroy data
    nobody can get back. So: copy to a sibling temporary first and swap only
    once the copy has finished, rather than moving the old file aside and
    hoping the copy works.

    The .wal sidecar is part of the database. An ingest that was interrupted,
    or that simply has not checkpointed yet, keeps its most recent work there,
    and <name>.duckdb on its own then opens with none of it -- often with no
    tables at all. So the log is copied with the database. Any *stale* log at
    the destination is moved aside first, because DuckDB replays whatever
    <name>.duckdb.wal it finds next to the file it opens, and replaying one
    database's log into another is what produces
    'Catalog Error: Table with name "raw" already exists!'.
    """
    tmp, tmp_wal = dst + ".incoming", None
    try:
        shutil.copy2(src, tmp)
        make_writable(tmp)
        if os.path.exists(src + ".wal"):
            tmp_wal = tmp + ".wal"
            shutil.copy2(src + ".wal", tmp_wal)
            make_writable(tmp_wal)
    except BaseException:
        # Includes Ctrl+C halfway through copying gigabytes.
        discard(tmp, tmp_wal)
        raise
    kept = []
    try:
        for path in (dst, dst + ".wal"):
            if os.path.exists(path):
                # os.replace overwrites an existing .bak in one step. shutil.move
                # would fall back to a full byte copy on Windows, where renaming
                # onto an existing file raises instead of replacing it.
                os.replace(path, path + ".bak")
                kept.append(path)
        os.replace(tmp, dst)
        if tmp_wal:
            os.replace(tmp_wal, dst + ".wal")
    except OSError:
        # On Windows a file another process holds open cannot be renamed at
        # all, so the swap can fail with the old database already moved aside.
        # Put it back and drop the copy: a failed install has to leave the
        # destination exactly as it found it, not missing.
        for path in kept:
            if not os.path.exists(path):
                os.replace(path + ".bak", path)
        discard(tmp, tmp_wal)
        raise
    kept = [os.path.basename(p) for p in kept]
    note = ""
    if kept:
        note = "\n    kept the previous " + ", ".join(f"{k} as {k}.bak"
                                                      for k in kept)
    return (f"  installed {os.path.basename(dst)}"
            + (" (with its write-ahead log)" if tmp_wal else "") + note)


def apply_prebuilt(src, dst, kind, action):
    """Put one prebuilt database into place. -> True if dst gained its data."""
    name, target = os.path.basename(src), os.path.basename(dst)
    if action == "already":
        print(f"  {name} is already installed as {target}; nothing to do")
        return True
    if action == "skip":
        print(f"  {name}: NOT loaded. {target} already holds {kind} data and "
              f"{kind} data cannot be merged, so this file would have replaced "
              "it.\n"
              f"    To use this one instead, move it somewhere on its own and "
              f"run: python atlas.py get \"{src}\"")
        return False
    if action == "merge":
        added, skipped, problem = merge_db(kind, src, dst)
        if problem:
            print(f"  {name}: could NOT be merged into {target}: {problem}")
            return False
        if added:
            print(f"  merged {len(added)} new item(s) from {name} into {target}"
                  + (f", {len(skipped)} already there" if skipped else ""))
        else:
            print(f"  {name}: nothing new; all {len(skipped)} item(s) are "
                  f"already in {target}")
        return True
    if same_file(src, dst):
        print(f"  {name} is already installed as {target}; nothing to do")
        return True
    try:
        print(install_db(src, dst))
    except (OSError, shutil.Error) as e:                    # noqa: BLE001
        # A file another process holds open (on Windows an open DuckDB handle
        # locks the file outright), a full disk, a read-only destination, a
        # source that disappeared mid-copy. Every one of these used to end the
        # run in a traceback -- and, before this, after the previous database
        # had already been moved out of the way.
        print(f"  {name}: could NOT be installed as {target}: {e}")
        return False
    return True


def ingest_targets(found):
    """Which databases the planned ingests are about to write into."""
    return {"iq": "iq", "psd": "psd", "pfp": "pfp", "csv": "spectrum"}


def unusable_targets(found):
    """-> [(path, why)] for databases an ingest would write but cannot read.

    DuckDB opens the destination read-write, so pointing an ingest at a corrupt
    or foreign file used to mean one raw IOException traceback per ingest script
    and no statement of the cause.
    """
    bad = []
    for key, kind in ingest_targets(found).items():
        if not found.get(key):
            continue
        p = db_path(kind)
        if not os.path.exists(p):
            continue
        real, why, _rows = identify_db(p)
        if real != kind:
            bad.append((p, why))
    return bad


def plan_for(root, found, compact):
    """-> (list of (label, argv), prebuilt actions, unreadable databases)."""
    import psd_ingest
    import pfp_ingest
    steps, (prebuilt, rejected) = [], prebuilt_plan(found)

    for d in sorted(found["iq"]):
        named = d if os.path.isdir(d) else os.path.dirname(d)
        steps.append((f"IQ captures in {short(d)}",
                      script("iq_ingest.py", d, "--dataset",
                             os.path.basename(os.path.normpath(named)))))
    for d in sorted(found["psd"]):
        for s in sorted(psd_ingest.discover(d, "max")):
            steps.append((f"CBRS PSD, sensor {s}, under {short(d)}",
                          script("psd_ingest.py", s, "--root", d)))
    for d in sorted(found["pfp"]):
        for s in sorted(pfp_ingest.discover(d, "max_peak")):
            steps.append((f"CBRS PFP, sensor {s}, under {short(d)}",
                          script("pfp_ingest.py", s, "--root", d)))
    # Summaries CSVs are whatever is left over that is still a CSV.
    for d in sorted(found["csv"]):
        steps.append((f"CBRS Summaries in {short(d)}",
                      script("build_db.py", "--csv-dir", d)))
    if compact and any(k for k in ("psd", "pfp") if found[k]):
        steps.append(("compact the CBRS databases", script("compact_db.py")))
    return steps, prebuilt, rejected


# ---- identifying databases whatever they are called --------------------

# A database is identified by the tables inside it, never by its filename.
# Someone's spectrum database may well be called spectrum_viewer.db.
DB_SIGNATURES = [
    ("iq", {"iq_stft"}),
    ("psd", {"psd_chunk"}), ("psd", {"psd"}),
    ("pfp", {"pfp_chunk"}), ("pfp", {"pfp"}),
    ("spectrum", {"raw"}),
]
DB_EXT = (".duckdb", ".db")

# The columns each of those tables must actually have. Matching on the table
# name alone meant a file holding an iq_stft of unrelated columns was reported
# as a healthy STFT pyramid -- "ready, start the viewer" -- and only failed at
# render time, by which point nothing said why. A superset is fine, so a later
# schema that adds a column still identifies.
TABLE_COLS = {
    "iq_stft": {"id", "level", "col0", "ncols", "chunk"},
    "psd": {"sensor", "t", "spec"},
    "psd_chunk": {"sensor", "t0", "t1", "n", "times", "specs"},
    "pfp": {"sensor", "freq", "t", "frame"},
    "pfp_chunk": {"sensor", "freq", "t0", "t1", "n", "times", "frames"},
    "raw": {"sensor", "freq", "t", "mx", "md", "mn"},
}


# What it takes to fold one database into another of the same kind: the column
# that identifies an independent unit of data, and every table keyed by it. The
# first table listed is the index -- the one that says which units exist -- and
# is always written by the matching ingest. Both the compacted and uncompacted
# table names appear; whichever is present in both files gets merged.
#
# spectrum is deliberately absent. Its pyramid levels are aggregates over the
# whole of `raw`, so two files cannot be combined without recomputing them,
# which is build_db.py's job rather than a copy.
MERGEABLE = {
    "iq": ("id", ("iq_meta", "iq_stft")),
    "psd": ("sensor", ("psd_meta", "psd", "psd_chunk")),
    "pfp": ("sensor", ("pfp_meta", "pfp", "pfp_chunk")),
}


def sql_str(s):
    """A path as a SQL literal. ATTACH takes no parameters, so it needs this."""
    return "'" + s.replace("'", "''") + "'"


def table_cols(con, catalog):
    """-> {table: [column, ...]} for one attached database, in column order."""
    out = {}
    for t, c in con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_catalog = ? AND table_schema = 'main' "
            "ORDER BY table_name, ordinal_position", [catalog]).fetchall():
        out.setdefault(t, []).append(c)
    return out


def merge_db(kind, src, dst):
    """Add everything in src that dst does not already have.

    -> (added, skipped, problem). added and skipped are unit keys (capture ids,
    or sensor names); problem is None on success, else one sentence saying why
    nothing was merged. src is opened read-only and never modified, and a unit
    already in dst is left alone rather than duplicated, so this is idempotent.

    Merging rather than copying is the whole point: two databases of the same
    kind used to mean the second one silently replacing the first.
    """
    import duckdb
    key, tables = MERGEABLE[kind]
    index = tables[0]
    con = None
    try:
        con = duckdb.connect(dst)
        con.execute(f"ATTACH {sql_str(os.path.abspath(src))} AS m (READ_ONLY)")
        here = table_cols(con, con.execute(
            "SELECT current_database()").fetchone()[0])
        there = table_cols(con, "m")
        if index not in there or index not in here:
            missing = os.path.basename(src if index not in there else dst)
            return [], [], f"{missing} has no {index} table to merge on"
        # Validate every shared table before writing any of it, so a mismatch
        # cannot leave half a merge behind.
        shared = [t for t in tables if t in here and t in there]
        for t in shared:
            if here[t] != there[t]:
                return [], [], (f"table {t} has different columns in the two "
                                f"files ({', '.join(there[t])} vs "
                                f"{', '.join(here[t])})")
        mine = {r[0] for r in con.execute(
            f"SELECT DISTINCT {key} FROM {index}").fetchall()}
        theirs = [r[0] for r in con.execute(
            f"SELECT DISTINCT {key} FROM m.{index} ORDER BY 1").fetchall()]
        add = [k for k in theirs if k not in mine]
        skip = [k for k in theirs if k in mine]
        if not add:
            return [], skip, None
        holes = ",".join("?" * len(add))
        for t in shared:
            cols = ", ".join(here[t])
            con.execute(f"INSERT INTO {t} ({cols}) SELECT {cols} FROM m.{t} "
                        f"WHERE {key} IN ({holes})", add)
        return add, skip, None
    except Exception as e:                                  # noqa: BLE001
        return [], [], str(e).splitlines()[0][:120]
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:                               # noqa: BLE001
                pass


def identify_db(path):
    """Open any database file and say what it actually is.

    -> (name, schema_label, rows) when it is one of ours,
       (None, why_not, 0) otherwise. Never raises.
    """
    import duckdb
    try:
        c = duckdb.connect(path, read_only=True)
    except Exception as e:                                  # noqa: BLE001
        why = str(e).splitlines()[0].replace(path, os.path.basename(path))
        # Long enough for DuckDB's own sentence, which names the cause;
        # cutting at 70 chopped it mid-word.
        return None, f"not readable as a database ({why[:120]})", 0
    try:
        cols = table_cols(c, c.execute("SELECT current_database()").fetchone()[0])
        tables = set(cols)
        wrong = []
        for name, need in DB_SIGNATURES:
            if not need <= tables:
                continue
            table = sorted(need)[0]
            want = TABLE_COLS.get(table, set())
            have = set(cols.get(table, []))
            if not want <= have:
                wrong.append(f"{table} is missing "
                             + ", ".join(sorted(want - have)))
                continue
            label = next((lbl for t, lbl in SCHEMAS[name] if t in tables),
                         "unknown")
            rows = c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            return name, label, rows
        if wrong:
            # The tables are named like ours but do not hold what we read.
            return None, f"wrong schema for an ATLAS database ({wrong[0]})", 0
        listed = ", ".join(sorted(tables)[:4]) or "no tables"
        return None, f"not an ATLAS database (contains: {listed})", 0
    except Exception as e:                                  # noqa: BLE001
        return None, f"unreadable ({str(e).splitlines()[0][:70]})", 0
    finally:
        try:
            c.close()
        except Exception:
            pass


# One size formatter for the whole project; fetch.py owns it because that is
# the lowest layer that needs one.
from fetch import fmt_size as fmt_bytes             # noqa: E402


def dir_size(root, cap=400000):
    """Total bytes under root, best effort."""
    total = n = 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
            n += 1
            if n >= cap:
                return total
    return total


def free_space(where=None):
    try:
        return shutil.disk_usage(where or HERE).free
    except OSError:
        return None


def pkg_version(dist):
    """Installed version of a distribution. Reading __version__ off the module
    is deprecated in several of these packages, so ask the metadata instead."""
    try:
        import importlib.metadata as md
        return md.version(dist)
    except Exception:                                       # noqa: BLE001
        return ""


def env_kind():
    """-> (kind, label). Which Python environment this is running in."""
    if os.environ.get("CONDA_PREFIX"):
        return "conda", os.environ.get("CONDA_DEFAULT_ENV") or os.environ["CONDA_PREFIX"]
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "venv", sys.prefix
    return "system", sys.prefix


# The four packages nothing works without, in one place. This used to be spelled
# out separately in doctor, in the no-argument path and in setup, and the other
# three subcommands checked nothing at all -- so `status`, `scan`, `get` and
# `serve` each died on a raw ModuleNotFoundError from whichever module happened
# to import duckdb first, naming a Python module instead of the thing to run.
CORE_DEPS = (("duckdb", "duckdb"), ("numpy", "numpy"),
             ("flask", "Flask"), ("PIL", "Pillow"))
MIN_PY = (3, 10)


def missing_deps():
    out = []
    for mod, pkg in CORE_DEPS:
        try:
            __import__(mod)
        except Exception:                                   # noqa: BLE001
            out.append(pkg)
    return out


def require_python():
    """None if this interpreter is new enough, else why it is not."""
    if sys.version_info >= MIN_PY:
        return None
    v = sys.version_info
    return (f"Python {v.major}.{v.minor}.{v.micro} is too old: ATLAS needs "
            f"{MIN_PY[0]}.{MIN_PY[1]} or newer (duckdb has no wheels below "
            f"that).\n  Installing dependencies will not help. Use a newer "
            f"interpreter:\n    py -3.12 atlas.py ...        # Windows\n"
            f"    python3.12 atlas.py ...      # macOS / Linux\n"
            f"  This one is {sys.executable}")


def require_deps():
    """None when everything imports, else one actionable message."""
    bad = require_python()
    if bad:
        return bad
    missing = missing_deps()
    if not missing:
        return None
    return f"Missing dependencies: {', '.join(missing)}\n\n{install_hint()}"


def install_hint():
    """The exact install command for the environment actually in use."""
    kind, label = env_kind()
    if kind == "conda":
        return (f"conda environment '{label}' is active. Install into it with:\n"
                f"    python -m pip install -r requirements.txt")
    if kind == "venv":
        return ("a virtual environment is active. Install into it with:\n"
                "    python -m pip install -r requirements.txt")
    return ("no virtual environment is active, and a bare pip install fails on\n"
            "  most Linux and macOS systems. Make one first:\n"
            "    python -m venv .venv\n"
            "    source .venv/bin/activate      # Windows: .venv\\Scripts\\activate\n"
            "    python -m pip install -r requirements.txt\n"
            "  Or with conda/miniforge:\n"
            "    conda create -n atlas python=3.12 -y && conda activate atlas\n"
            "    python -m pip install -r requirements.txt")


# ---- searching for data you already have ------------------------------

def default_roots():
    """Where a spectrum download plausibly landed on someone's machine."""
    home = os.path.expanduser("~")
    cand = [os.environ.get("SEA_DATA_ROOT"),
            os.path.join(HERE, "SEA-DATA"), os.path.join(HERE, "iqdata"),
            os.path.join(HERE, "downloads"), os.path.join(HERE, "data"),
            os.path.join(home, "Downloads"), os.path.join(home, "Desktop"),
            os.path.join(home, "Documents")]
    out = []
    for p in cand:
        if p and os.path.isdir(p) and os.path.abspath(p) not in out:
            out.append(os.path.abspath(p))
    return out


def deep_roots():
    """Everything in default_roots, plus the whole home folder and, on
    Windows, every other fixed drive. Slow on purpose."""
    out = list(default_roots())
    home = os.path.abspath(os.path.expanduser("~"))
    if home not in out:
        out.append(home)
    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if letter != "C" and os.path.isdir(drive):
                out.append(drive)
    return out


def survey(root, max_files, max_depth=12):
    """Inventory one directory: what is recognised, and what is not."""
    import collections
    found = {"iq": 0, "psd": 0, "pfp": 0, "summaries": 0, "duckdb": []}
    sensors, days, unknown = set(), set(), collections.Counter()
    psd_re, pfp_re = name_res()
    files = 0
    gen, limits = walk_limited(root, max_files=max_files,
                               max_depth=max_depth)
    for dirpath, name in gen:
        files += 1
        kind = kind_of(dirpath, name)
        if kind == "duckdb":
            found["duckdb"].append(os.path.join(dirpath, name))
        elif kind in ("psd", "pfp"):
            found[kind] += 1
            m = (psd_re if kind == "psd" else pfp_re).match(name)
            sensors.add(m.group("sensor"))
            days.add(m.group("day"))
        elif kind:
            found[kind] += 1
        elif not name.lower().endswith(COMPANION_EXT):
            unknown[os.path.splitext(name)[1].lower() or "(no extension)"] += 1
    return {"root": root, "found": found, "sensors": sorted(sensors),
            "days": sorted(days), "unknown": unknown, "files": files,
            "limits": limits, "max_depth": max_depth}


def report(s):
    """Print one directory's inventory. Returns True if anything was usable."""
    f, any_hit = s["found"], False
    print(f"\n  {s['root']}")
    if f["iq"]:
        print(f"      {f['iq']:>7} IQ capture file(s)")
        any_hit = True
    for kind, label in (("psd", "CBRS PSD export(s)"),
                        ("pfp", "CBRS PFP export(s)")):
        if f[kind]:
            span = (f"{s['days'][0]} .. {s['days'][-1]}" if s["days"] else "")
            print(f"      {f[kind]:>7} {label:<22} "
                  f"{len(s['sensors'])} sensor(s)  {span}")
            any_hit = True
    if f["summaries"]:
        print(f"      {f['summaries']:>7} CBRS Summaries CSV(s)")
        any_hit = True
    for p in f["duckdb"]:
        stem = os.path.basename(p)[:-len(".duckdb")].rstrip("_c")
        mark = "prebuilt database" if stem in DBS else "database (not one of ours)"
        print(f"      {'':>7} {mark}: {os.path.basename(p)}")
        any_hit = any_hit or stem in DBS
    if s["sensors"]:
        print(f"      sensors: {', '.join(s['sensors'])}")
    unknown = sum(s["unknown"].values())
    if unknown:
        top = ", ".join(f"{ext} {n}" for ext, n in s["unknown"].most_common(6))
        print(f"      {unknown:>7} file(s) not recognised  ({top})")
    if not any_hit and not unknown:
        print("           nothing here")
    if s["limits"]["files"]:
        print("      NOTE: hit the file limit; re-run with a bigger --max-files "
              "or point --root at a narrower folder")
    if s["limits"]["depth"]:
        print(f"      NOTE: stopped at {s['max_depth']} directory levels below "
              f"this folder; {s['limits']['depth']} deeper folder(s) were NOT "
              "searched.")
        print("            Re-run with --max-depth N, or name the deeper "
              "folder directly.")
    return any_hit


def scan_depth(args):
    """How many directory levels to search.

    --deep widens this as well as adding roots. It used to add roots only, so
    "search wider" still stopped at 12 levels and missed anything below that
    without saying so.
    """
    if getattr(args, "max_depth", None) is not None:
        return args.max_depth
    return 24 if getattr(args, "deep", False) else 12


def cmd_scan(args):
    """Find spectrum data anywhere on this machine and say what it is."""
    roots = args.roots or (deep_roots() if args.deep else default_roots())
    if not roots:
        example = ("C:\\Users\\you\\Downloads" if os.name == "nt"
                   else "~/Downloads")
        print("No obvious place to look. Pass a folder: "
              f"python atlas.py scan {example}", file=sys.stderr)
        return 1
    print(f"searching {len(roots)} place(s), up to {args.max_files:,} files each."
          + ("" if args.deep else "  Add --deep to search wider, or name a "
                                  "folder to search exactly that."))
    missing = [r for r in roots if not os.path.isdir(r)]
    roots = [r for r in roots if os.path.isdir(r)]
    for r in roots:
        print(f"  - {r}")
    for r in missing:
        print(f"  - {r}   (does not exist, skipped)")
    if not roots:
        print("\nNone of those folders exist.", file=sys.stderr)
        return 1

    hits, empty, csv_missed, stopped_deep = [], [], 0, 0
    for r in roots:
        s = survey(r, args.max_files, scan_depth(args))
        if report(s):
            hits.append(s)
        else:
            empty.append(r)
        csv_missed += s["unknown"].get(".csv", 0)
        stopped_deep += s["limits"]["depth"]

    print()
    if csv_missed:
        # The most likely way this misses real data: exports named differently.
        print(f"{csv_missed} .csv file(s) were found but not recognised. CBRS "
              "exports are matched by name:")
        print("    PSD        <YYYY-MM-DD>_<sensor>_<stat>.csv")
        print("    PFP        PFP_<YYYY-MM-DD>_<sensor>_<stat>.csv")
        print("    Summaries  any .csv whose header has sensor_name and "
              "channel_frequency_mhz")
        print("  If yours look different, rename them to match or say so and "
              "the pattern can be widened.\n")
    if not hits:
        if stopped_deep:
            # Say this first: it is a concrete, fixable reason the search came
            # up empty, unlike the generic advice below.
            print(f"The search stopped at {scan_depth(args)} directory levels "
                  f"and did not look inside {stopped_deep} deeper folder(s).")
            print("  Re-run with --max-depth 40, or name the folder that holds "
                  "the data directly.")
            return 1
        print("Found nothing to ingest. Either the data is somewhere not "
              "searched (pass the folder directly, or use --deep), or it is "
              "named differently than the patterns above.")
        return 1
    print("To ingest what was found, run:")
    for s in hits:
        print(f'    python atlas.py get "{s["root"]}"')
    if empty:
        print(f"\n(nothing in: {', '.join(empty)})")
    return 0


# ---- doctor: the one command that checks and fixes everything ----------

class Report:
    """Collects PASS / WARN / FAIL lines so the verdict reflects all of them."""

    def __init__(self):
        self.bad = self.warn = 0

    def ok(self, msg, detail=""):
        print(f"  [ OK ] {msg}" + (f"  {detail}" if detail else ""))

    def warned(self, msg, detail=""):
        self.warn += 1
        print(f"  [WARN] {msg}" + (f"  {detail}" if detail else ""))

    def failed(self, msg, detail=""):
        self.bad += 1
        print(f"  [FAIL] {msg}" + (f"  {detail}" if detail else ""))


def doctor_env(r):
    print("\n1. Python and environment")
    v = sys.version_info
    if v >= (3, 10):
        r.ok(f"Python {v.major}.{v.minor}.{v.micro}", sys.executable)
    else:
        r.failed(f"Python {v.major}.{v.minor} is too old; 3.10 or newer is needed")
    kind, label = env_kind()
    if kind == "conda":
        r.ok(f"conda environment active: {label}")
    elif kind == "venv":
        r.ok(f"virtual environment active: {label}")
    else:
        r.warned("no virtual environment active",
                 "installs may fail with externally-managed-environment")
    if os.access(HERE, os.W_OK):
        r.ok("repository directory is writable", HERE)
    else:
        r.failed("repository directory is NOT writable", HERE)


def doctor_deps(r):
    print("\n2. Dependencies")
    need = [("duckdb", "duckdb"), ("numpy", "numpy"),
            ("flask", "Flask"), ("PIL", "Pillow")]
    missing = []
    for mod, pkg in need:
        try:
            __import__(mod)
            r.ok(f"import {mod}", pkg_version(pkg))
        except Exception as e:                              # noqa: BLE001
            missing.append(pkg)
            r.failed(f"import {mod}", str(e).splitlines()[0][:60])
    for mod, pkg, why in [("waitress", "waitress", "faster server, optional"),
                          ("nptdms", "npTDMS", "only for NI .tdms captures")]:
        try:
            __import__(mod)
            r.ok(f"import {mod}", why)
        except Exception:                                   # noqa: BLE001
            print(f"  [ -- ] {mod} not installed ({why})")
    if missing:
        print(f"\n  To fix: {install_hint()}")
    return not missing


def doctor_disk(r):
    print("\n3. Disk space")
    free = free_space()
    if free is None:
        r.warned("could not read free space")
        return
    used = sum(os.path.getsize(db_path(n)) for n in DBS
               if os.path.exists(db_path(n)))
    detail = f"{fmt_bytes(free)} free" + (f", {fmt_bytes(used)} already in databases"
                                          if used else "")
    if free < 200 * 1024 ** 2:
        r.failed("less than 200 MB free; even the demo may not build", detail)
    elif free < 2 * 1024 ** 3:
        r.warned("under 2 GB free", detail + ". The demo fits; real CBRS "
                 "databases are multi-GB and will not.")
    else:
        r.ok("enough space for the demo and then some", detail)


def doctor_databases(r, adopt):
    """Report every database beside serve.py, including oddly-named ones."""
    print("\n4. Databases")
    canonical = {os.path.abspath(db_path(n)): n for n in DBS}
    seen = {}
    for n in DBS:
        p = db_path(n)
        if not os.path.exists(p):
            continue
        real, label, rows = identify_db(p)
        if real is None:
            r.failed(f"{os.path.basename(p)} is unusable", label)
        elif real != n:
            r.failed(f"{os.path.basename(p)} actually contains {real} data", label)
        else:
            seen[n] = rows
            r.ok(f"{os.path.basename(p)}: {label}", f"{rows:,} rows")
    if not seen:
        print("  [ -- ] no ATLAS databases built yet")

    # Anything else database-shaped sitting in the repo: spectrum_viewer.db and
    # friends. Identify by schema, then offer to put it where serve.py looks.
    strays = []
    where = db_dir()
    for name in sorted(os.listdir(where) if os.path.isdir(where) else []):
        p = os.path.abspath(os.path.join(where, name))
        if not os.path.isfile(p) or not name.lower().endswith(DB_EXT):
            continue
        if p in canonical:
            continue
        real, label, rows = identify_db(p)
        if real:
            strays.append((p, real, label, rows))
            print(f"  [ !! ] {name} holds {real} data ({label}, {rows:,} rows) "
                  f"but serve.py looks for {real}.duckdb")
        else:
            print(f"  [ -- ] {name} ignored: {label}")
    for p, real, label, rows in strays:
        dst = db_path(real)
        if not adopt:
            r.warned(f"{os.path.basename(p)} is not being used",
                     f"re-run with --adopt to load it as {real}.duckdb")
            continue
        if real in seen and real not in MERGEABLE:
            # Adopting here would copy this file over a database that already
            # holds data of the same kind, and spectrum data cannot be merged.
            # --adopt asks for a stray to be put to use, not for the working
            # database to be replaced by it.
            r.warned(f"{os.path.basename(p)} was NOT adopted",
                     f"{real}.duckdb already holds {real} data, which cannot "
                     f"be merged. Move {real}.duckdb aside first if you mean "
                     "to replace it.")
            continue
        # Merge when there is already data to merge into, so adopting a second
        # database adds to the library instead of replacing it.
        action = "merge" if real in seen else "install"
        if apply_prebuilt(p, dst, real, action):
            r.ok(f"adopted {os.path.basename(p)} as {real}.duckdb",
                 f"{rows:,} rows")
            seen[real] = max(seen.get(real, 0), rows)
        else:
            r.failed(f"could not adopt {os.path.basename(p)}")
    return seen


def doctor_data(r, args):
    print("\n5. Spectrum data on this machine")
    roots = args.roots or (deep_roots() if args.deep else default_roots())
    roots = [x for x in roots if os.path.isdir(x)]
    if not roots:
        print("  [ -- ] none of the usual folders exist")
        return []
    hits = []
    for root in roots:
        s = survey(root, args.max_files, scan_depth(args))
        f = s["found"]
        total = f["iq"] + f["psd"] + f["pfp"] + f["summaries"]
        if total:
            size = dir_size(root)
            kinds = ", ".join(f"{f[k]} {k}" for k in
                              ("iq", "psd", "pfp", "summaries") if f[k])
            r.ok(f"{short(root)}", f"{kinds}  ({fmt_bytes(size)} of source files)")
            hits.append((root, size))
        if s["unknown"].get(".csv"):
            r.warned(f"{s['unknown']['.csv']} unrecognised .csv under {short(root)}",
                     "names may not match the expected patterns; run "
                     "'atlas.py scan' for the details")
    if not hits:
        print("  [ -- ] no source data found (that is fine; the demo needs none)")
    return hits


def doctor_verify(r):
    print("\n7. End-to-end verification")
    rc = subprocess.run([sys.executable,
                         os.path.join(HERE, "examples", "verify.py")],
                        cwd=HERE, capture_output=True, text=True)
    out = rc.stdout + rc.stderr
    shown = 0
    for line in out.splitlines():
        if line.strip().startswith(("[PASS]", "[FAIL]", "RESULT:")):
            print(f"  {line.strip()}")
            shown += 1
    if rc.returncode == 0:
        r.ok("every endpoint the viewer needs responded")
        return True
    # A crash produces none of those lines, so claiming "the output above names
    # the check" was sometimes false. Surface the exception itself instead.
    if any("[FAIL]" in l for l in out.splitlines()):
        r.failed("verification failed; the check that failed is named above")
    else:
        r.failed("verification crashed before it could finish",
                 last_line(out))
        if not shown:
            print("         re-run it directly for the whole traceback:")
            print(f"           {short(sys.executable)} "
                  f"{short(os.path.join(HERE, 'examples', 'verify.py'))}")
    return False


def cmd_doctor(args):
    """Check everything, fix what is safe to fix, and say what to do next."""
    print("ATLAS doctor" + ("  (dry run, nothing will change)" if args.dry_run else ""))
    print(f"repository: {HERE}")
    r = Report()

    doctor_env(r)
    deps_ok = doctor_deps(r)
    if not deps_ok:
        print("\nVERDICT: dependencies are missing. Install them with the "
              "command above, then run this again.")
        return 1
    doctor_disk(r)
    built = doctor_databases(r, adopt=args.adopt and not args.dry_run)
    hits = doctor_data(r, args)

    print("\n6. Demo data")
    iq = db_path("iq")
    if built.get("iq"):
        r.ok("an IQ database already exists; leaving it alone")
    elif os.path.exists(iq):
        # There is a file at iq.duckdb but section 4 could not read it as one
        # of ours. Building the demo would write new tables straight into
        # whatever it actually is -- someone else's database, or a corrupt
        # download -- with no backup and no mention. Never do that.
        r.failed(f"not building the demo: {os.path.basename(iq)} already "
                 "exists and is not a usable ATLAS database",
                 "move or delete that file first, then run this again")
    elif args.dry_run:
        print("  [ -- ] would build the synthetic demo")
    else:
        print("  building the synthetic demo so there is something to look at ...")
        # --iq, because the only file this section checked before deciding to
        # write is iq.duckdb. make_sample.py also builds the three CBRS
        # databases, and creating those here would put demo sensors next to
        # real data -- or on top of a legacy database still waiting to be
        # adopted -- on the strength of a check that never looked at them.
        rc = subprocess.run([sys.executable,
                             os.path.join(HERE, "examples", "make_sample.py"),
                             "--iq"],
                            cwd=HERE, capture_output=True, text=True)
        if rc.returncode == 0:
            r.ok("demo capture built")
            built["iq"] = 1
        else:
            r.failed("could not build the demo", last_line(rc.stdout + rc.stderr))

    verified = doctor_verify(r) if not args.dry_run else True

    print("\n" + "=" * 62)
    if r.bad:
        print(f"VERDICT: {r.bad} problem(s), {r.warn} warning(s). Fix the FAIL "
              "lines above and run this again.")
        return 1
    print(f"VERDICT: working. {r.warn} warning(s)." if r.warn
          else "VERDICT: working, no problems found.")
    print("\n    python atlas.py serve        # then open http://127.0.0.1:8090")
    if hits:
        print("\nSource data was found but not ingested. To load it:")
        for root, size in hits:
            free = free_space()
            warn = ""
            if free is not None and free < size * 2:
                warn = (f"   # WARNING: {fmt_bytes(size)} of source but only "
                        f"{fmt_bytes(free)} free")
            print(f'    python atlas.py get "{root}"{warn}')
    return 0 if verified else 1


# ---- the no-argument path: work out the situation and offer the fix ----

def assess(max_files=60000):
    """One pass over everything that decides what to do next.

    Deliberately quiet: this runs before the user has asked for anything, so
    it reports nothing and just returns what it learned.
    """
    state = {"missing": missing_deps(), "built": {}, "strays": [], "data": [],
             "free": None}
    if state["missing"]:
        return state                        # nothing else can be trusted yet

    state["free"] = free_space()
    for n in DBS:
        p = db_path(n)
        if os.path.exists(p):
            real, label, rows = identify_db(p)
            if real == n and rows:
                state["built"][n] = (label, rows)
    where = db_dir()
    canonical = {os.path.abspath(db_path(n)) for n in DBS}
    for name in sorted(os.listdir(where) if os.path.isdir(where) else []):
        p = os.path.abspath(os.path.join(where, name))
        if os.path.isfile(p) and name.lower().endswith(DB_EXT) \
                and p not in canonical:
            real, _label, rows = identify_db(p)
            if real:
                state["strays"].append((p, real, rows))
    for root in default_roots():
        s = survey(root, max_files)
        f = s["found"]
        if f["iq"] + f["psd"] + f["pfp"] + f["summaries"]:
            state["data"].append((root, f, dir_size(root)))
    return state


def describe(state):
    """The one-screen summary shown before the menu."""
    if state["built"]:
        for n, (label, rows) in state["built"].items():
            print(f"  ready    {n}: {label}, {rows:,} rows")
    else:
        print("  ready    nothing built yet")
    for p, real, rows in state["strays"]:
        print(f"  found    {os.path.basename(p)} holds {real} data "
              f"({rows:,} rows) but is not installed")
    for root, f, size in state["data"]:
        kinds = ", ".join(f"{f[k]} {k}" for k in
                          ("iq", "psd", "pfp", "summaries") if f[k])
        print(f"  found    {kinds} in {short(root)}  ({fmt_bytes(size)})")
    if state["free"] is not None:
        print(f"  disk     {fmt_bytes(state['free'])} free")


def ns(**kw):
    """An argparse-shaped object, for calling the subcommands directly."""
    base = dict(dry_run=False, adopt=False, deep=False, max_files=200000,
                max_depth=None,
                roots=None, compact=False, ask=False, filter=None, dest=None,
                fetch_args=[], target=None)
    base.update(kw)
    return argparse.Namespace(**base)


def build_menu(state):
    """-> list of (label, action). The first entry is the recommendation."""
    items = []

    for p, real, rows in state["strays"]:
        items.append((f"Install {os.path.basename(p)} as {real}.duckdb "
                      f"({rows:,} rows)",
                      lambda: cmd_doctor(ns(adopt=True))))
    for root, f, size in state["data"]:
        items.append((f"Load the data in {short(root)} ({fmt_bytes(size)})",
                      lambda r=root: cmd_get(ns(target=r))))
    if state["built"]:
        items.append(("Start the viewer", lambda: cmd_serve(ns())))
    else:
        items.append(("Build the demo and start the viewer",
                      lambda: (cmd_setup(ns()), cmd_serve(ns()))[-1]))

    names = [k for k in load_datasets() if not k.startswith("_")]
    if names:
        items.append((f"Download a dataset from NIST ({', '.join(names)})",
                      lambda: choose_dataset(names)))
    items.append(("Search this whole machine for data",
                  lambda: cmd_scan(ns(deep=True))))
    items.append(("Check everything (environment, disk, databases)",
                  lambda: cmd_doctor(ns())))
    return items


def choose_dataset(names):
    data = load_datasets()
    print("\nWhich dataset?")
    for i, n in enumerate(names, 1):
        entry = data[n]
        print(f"  {i}) {n:<12} {entry.get('title', '')}")
        if entry.get("note"):
            print(f"     {entry['note']}")
    pick = prompt(len(names))
    if pick is None or pick is NO_INPUT:
        return 0
    name = names[pick]
    entry = data[name]
    sample = entry.get("sample", 3)
    # A dataset may name extra fetch.py flags it needs, e.g. --allow-host for a
    # record whose files are served from somewhere unexpected.
    flags = [str(f) for f in entry.get("flags", [])]

    # These records run to hundreds of gigabytes. Default to a slice small
    # enough that trying it out costs megabytes.
    sizes = [(f"Just a sample: the {sample} smallest captures (a few MB)",
              ns(target=name, fetch_args=["--first", str(sample)] + flags))]
    if entry.get("filter"):
        sizes.append((f"The usual slice (--filter {entry['filter']})",
                      ns(target=name, fetch_args=list(flags))))
    sizes.append(("Everything in the record (this may be very large)",
                  ns(target=name, filter="", fetch_args=list(flags))))
    print("\nHow much of it?")
    for i, (label, _) in enumerate(sizes, 1):
        print(f"  {i}) {label}" + ("      <- recommended" if i == 1 else ""))
    how = prompt(len(sizes))
    if how is None or how is NO_INPUT:
        return 0
    return cmd_get(sizes[how][1])


# Returned by prompt() when there is nothing to read, as opposed to the user
# choosing to quit. The two deserve different endings.
NO_INPUT = object()


def interactive():
    """True only when there is a person who can both see and answer a menu.

    stdin.isatty() alone is not enough: on Windows NUL is a character device,
    so a run with stdin redirected from NUL reports a terminal and then hits
    EOF on the first read. Requiring stdout to be a terminal as well means a
    redirected or captured run takes the non-interactive path, which is the
    one that prints advice instead of waiting.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:                                       # noqa: BLE001
        return False


def prompt(n):
    """Ask for a number in 1..n. -> index, None to quit, NO_INPUT at EOF."""
    while True:
        try:
            raw = input(f"\nChoose 1-{n} (or q to quit): ").strip().lower()
        except (EOFError, RuntimeError, OSError):
            # Redirected from empty, closed outright, or unreadable: all three
            # mean there is nothing to read, which is not the same as quitting.
            return NO_INPUT
        except KeyboardInterrupt:
            print()
            return None
        if raw in ("q", "quit", "exit", ""):
            return None
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        print(f"  not a choice. Enter a number from 1 to {n}, or q.")


def cmd_auto(args):
    """`python atlas.py` with no arguments: figure it out and offer the fix."""
    print("ATLAS")
    print("checking this machine ...")
    bad = require_deps()
    if bad:
        print(f"\n{bad}")
        return 1
    state = assess()

    print()
    describe(state)
    problem = db_dir_problem()
    if problem:
        print(f"\n{problem}")
        return 1

    items = build_menu(state)

    def advise():
        # No one to ask (a script, a CI job, a piped run). Say what the
        # recommended command is rather than guessing and doing it.
        print(f"\nRecommended: {items[0][0]}")
        print("Run 'python atlas.py doctor' for the full report, or "
              "'python atlas.py --help' for every command.")
        return 0

    if not interactive():
        return advise()

    print("\nWhat would you like to do?")
    for i, (label, _) in enumerate(items, 1):
        print(f"  {i}) {label}" + ("      <- recommended" if i == 1 else ""))
    pick = prompt(len(items))
    if pick is NO_INPUT:
        return advise()             # looked interactive, was not
    if pick is None:
        print("nothing done")
        return 0
    return items[pick][1]() or 0


# ---- subcommands ------------------------------------------------------

def db_size(path):
    """On-disk size of a database, counting an uncheckpointed write-ahead log.

    An ingest killed mid-transaction leaves its work in <name>.duckdb.wal, and
    atlas.py only ever opens read-only so it never gets checkpointed away.
    Reporting the main file alone showed "0.0 MB" for a database with megabytes
    of real data sitting beside it.
    """
    total = os.path.getsize(path)
    wal = path + ".wal"
    extra = os.path.getsize(wal) if os.path.exists(wal) else 0
    return f"{total / 1e6:.1f} MB" + (f" +{extra / 1e6:.1f} wal" if extra else "")


def cmd_status(args):
    import duckdb
    print(f"ATLAS in {HERE}")
    if os.path.abspath(db_dir()) != os.path.abspath(HERE):
        print(f"databases in {db_dir()}  (ATLAS_DB_DIR)")
    print()
    rows, present, broken = [], set(), []
    for name in DBS:
        p = db_path(name)
        if not os.path.exists(p):
            rows.append((name, "absent", "", "", ""))
            continue
        # identify_db is the single judge of what a file is -- it validates the
        # columns, not just the table names. Deciding that again here meant
        # status called a database with an iq_stft of unrelated columns a
        # healthy "stft pyramid, 1 rows" and told the user to start the viewer.
        real, why, n = identify_db(p)
        if real != name:
            broken.append((p, why if real is None else
                           f"holds {real} data, not {name} data"))
            rows.append((name, "UNUSABLE (see below)", "", "", db_size(p)))
            continue
        try:
            c = duckdb.connect(p, read_only=True)
            tables = {r[0] for r in c.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()}
            kind = next((lbl for t, lbl in SCHEMAS[name] if t in tables), "unknown")
            span = ""
            if name in ("psd", "pfp") and f"{name}_meta" in tables:
                r = c.execute(f"SELECT min(t_min), max(t_max) FROM {name}_meta").fetchone()
                if r and r[0] is not None:
                    span = f"{(r[1]-r[0])/86400:.1f} days"
                else:
                    kind += " (no time range: nothing ingested)"
            if name == "iq" and "iq_meta" in tables:
                span = f"{c.execute('SELECT count(*) FROM iq_meta').fetchone()[0]} capture(s)"
            c.close()
            if n:
                present.add(name)
            rows.append((name, kind, f"{n:,} rows", span, db_size(p)))
        except Exception as e:                              # noqa: BLE001
            # A DuckDB error runs to several hundred characters. Putting it in
            # the table padded every other column out to its width; it belongs
            # under the table, once.
            broken.append((p, str(e).splitlines()[0]))
            rows.append((name, "UNREADABLE (see below)", "", "", db_size(p)))
        # A file at a canonical path that is not what it claims is a problem,
        # not an absence: report it rather than leaving the row blank.

    w = min(34, max(len(r[1]) for r in rows) + 2)
    for name, kind, n, span, size in rows:
        print(f"  {name + '.duckdb':18} {kind:{w}} {n:>14}  {span:>14}  {size:>10}")

    for p, why in broken:
        print(f"\n  {os.path.basename(p)} is not usable:\n    {why[:220]}")
        print(f"    Move or delete {short(p)} and rebuild it.")

    print()
    if broken and not present:
        print("Nothing usable is built. Deal with the unreadable file(s) above "
              "first.")
        return 1
    if not present:
        print("Nothing is built yet. Start with:\n"
              "    python atlas.py doctor           # check everything, fix what it can\n"
              "    python atlas.py setup            # zero-download demo\n"
              "    python atlas.py scan             # find data already on this machine\n"
              "    python atlas.py get <folder>     # ingest a folder you have\n"
              "    python atlas.py get mds2-3177    # data from NIST PDR")
    else:
        if broken:
            print("Some layers are readable and some are not (see above); the "
                  "viewer will show only what works.")
        print("Ready. Start the viewer with:\n    python atlas.py serve")
        if {"psd", "pfp"} & present and not any(
                os.path.exists(db_path(n) + ".bak") for n in ("psd", "pfp")):
            print("\nOptional, once the CBRS databases get large:\n"
                  "    python ingest/compact_db.py      # smaller and faster")
    return 0


def cmd_setup(args):
    # install_hint() reads the environment actually in use, so it does not tell
    # someone already inside a virtualenv to make one, and it names
    # externally-managed-environment on the systems that raise it.
    bad = require_deps()
    if bad:
        print(bad, file=sys.stderr)
        return 1
    rc = run([sys.executable, os.path.join(HERE, "examples", "verify.py")],
             args.dry_run)
    if rc == 0 and not args.dry_run:
        print("\nSetup complete. Next:\n    python atlas.py serve")
    return rc


def cmd_serve(args):
    return run([sys.executable, os.path.join(HERE, "serve.py")], args.dry_run)


def cmd_get(args):
    target = args.target
    datasets = load_datasets()

    if target in datasets and not os.path.exists(target):
        entry = datasets[target]
        print(f"{target}: {entry.get('title', '')}")
        if entry.get("note"):
            print(f"  note: {entry['note']}")
        target = entry["record"]
        if entry.get("filter") and args.filter is None:
            args.filter = entry["filter"]
        # Any fetch flags the dataset needs, unless the caller already passed
        # them (the menu supplies them itself, so this must not double them up).
        args.fetch_args = list(args.fetch_args) + [
            str(f) for f in entry.get("flags", []) if str(f) not in args.fetch_args]

    # Already on disk? Then there is nothing to download.
    if os.path.exists(target):
        root = os.path.abspath(target)
        print(f"using local data: {root}")
    else:
        # Derive the download folder from the PARSED source, so a pasted
        # "<https://.../mds2-3177>" does not create a directory named
        # "mds2-3177>". fetch.py cleans the argument the same way.
        import fetch
        try:
            kind, val = fetch.parse_source(target)
            name = val if kind == "record" else os.path.basename(
                os.path.normpath(fetch.urllib.parse.urlsplit(val).path)) or "download"
        except Exception:                                   # noqa: BLE001
            name = os.path.basename(target.rstrip("/")) or "download"
        root = os.path.abspath(args.dest or os.path.join(download_dir(), name))
        argv = script("fetch.py", fetch.clean_source(target), "--dest", root)
        if args.filter:
            argv += ["--filter", args.filter]
        argv += args.fetch_args
        rc = run(argv, args.dry_run)
        if rc != 0:
            print("\ndownload did not finish; nothing was ingested",
                  file=sys.stderr)
            return rc
        if args.dry_run:
            print("\n(dry run: stopping here, since what to ingest depends on "
                  "what the download produces)")
            return 0

    found = classify(root)
    steps, prebuilt, rejected = plan_for(root, found, args.compact)
    if not steps and not prebuilt:
        if rejected:
            # There ARE databases here; they just cannot be read. Saying
            # "nothing recognisable" would send the user looking for the wrong
            # problem -- most often a download that finished corrupt.
            print(f"\n{len(rejected)} database file(s) under {root} could not "
                  "be read:", file=sys.stderr)
            for p, why in rejected[:5]:
                print(f"    {short(p)}\n      {why}", file=sys.stderr)
            print("  If one of these was downloaded, it may be corrupt: delete "
                  "it and re-run.", file=sys.stderr)
            return 1
        print(f"\nNothing recognisable under {root}.\n"
              "  Expected IQ captures (.sigmf-meta/.tdms/.npy/.complex1ch.float32),\n"
              "  CBRS PSD or PFP\n"
              "  CSV exports, Summaries CSVs, or prebuilt .duckdb files.",
              file=sys.stderr)
        if found["other"]:
            print(f"  {len(found['other'])} CSV file(s) are there but none has a "
                  "Summaries header (sensor_name, channel_frequency_mhz):",
                  file=sys.stderr)
            for f in found["other"][:5]:
                print(f"    {short(f)}", file=sys.stderr)
        return 1

    # An ingest opens its destination read-write, so check now rather than
    # letting each script die on its own DuckDB traceback.
    blocked = unusable_targets(found)
    if blocked and steps:
        print(f"\nCannot ingest: {len(blocked)} destination database(s) exist "
              "but are not usable ATLAS databases:", file=sys.stderr)
        for p, why in blocked:
            print(f"    {short(p)}\n      {why}", file=sys.stderr)
        print("  Move or delete them (or point ATLAS_DB_DIR elsewhere), then "
              "run this again.", file=sys.stderr)
        return 1

    print(f"\nplan for {root}:")
    if rejected:
        for p, why in rejected:
            print(f"  ! not loading {os.path.basename(p)}: {why}")
            print(f"      {short(p)}")
    for src, dst, kind, action in prebuilt:
        what = {"install": f"install prebuilt {os.path.basename(dst)}",
                "merge": f"merge {os.path.basename(src)} into "
                         f"{os.path.basename(dst)}",
                "already": f"{os.path.basename(src)} is already installed",
                "skip": f"skip {os.path.basename(src)} (see below)"}[action]
        print(f"  - {what}" + (" (no ingest needed)"
                               if action in ("install", "merge") else ""))
    for label, _ in steps:
        print(f"  - {label}")
    if found["other"]:
        # Always say what was passed over, so data named unexpectedly cannot be
        # skipped without the user seeing it.
        print(f"  ({len(found['other'])} CSV file(s) ignored, no Summaries "
              f"header; e.g. {short(found['other'][0])})")

    if args.ask and not args.dry_run:
        try:
            if input("\nproceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("nothing done")
                return 0
        # EOFError is stdin redirected from empty; RuntimeError is stdin closed
        # outright ("lost sys.stdin"); OSError is a descriptor that cannot be
        # read. All three mean the same thing: there is nobody to ask.
        except (EOFError, RuntimeError, OSError):
            print("\nno terminal to ask on; re-run without --ask")
            return 1

    failures = []
    for src, dst, kind, action in prebuilt:
        if args.dry_run:
            continue
        if not apply_prebuilt(src, dst, kind, action) and action != "skip":
            failures.append(f"loading {os.path.basename(src)}")

    for label, argv in steps:
        if run(argv, args.dry_run) != 0:
            failures.append(label)

    if args.dry_run:
        return 0
    print()
    if failures:
        print(f"{len(failures)} step(s) failed: " + "; ".join(failures),
              file=sys.stderr)
    cmd_status(args)
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="run 'python atlas.py get --help' for the download options")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("doctor", help="check EVERYTHING and fix what is safe "
                                     "(start here if anything is wrong)")
    p.add_argument("roots", nargs="*", default=None,
                   help="extra folders to search for data")
    p.add_argument("--deep", action="store_true",
                   help="search your home folder and every other drive too")
    p.add_argument("--adopt", action="store_true",
                   help="install oddly-named databases (spectrum_viewer.db and "
                        "the like) where serve.py looks for them")
    p.add_argument("--max-files", type=int, default=200000,
                   help="file budget per folder searched (default 200000)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="how many directory levels deep to search "
                        "(default 12, or 24 with --deep)")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("setup", help="check the install and build the demo")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("status", help="what is built, and what to run next")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("scan", help="search this machine for spectrum data")
    p.add_argument("roots", nargs="*", default=None,
                   help="folders to search (default: the usual download spots)")
    p.add_argument("--deep", action="store_true",
                   help="also search your home folder and every other drive")
    p.add_argument("--max-files", type=int, default=200000,
                   help="file budget per folder (default 200000)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="how many directory levels deep to search (default 12, "
                        "or 24 with --deep). Deeper folders are reported, never "
                        "skipped silently")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("serve", help="start the viewer")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("get", help="fetch if needed, then ingest whatever it is")
    p.add_argument("target", help="folder, file, dataset name, record id, or URL")
    p.add_argument("--dest", default=None,
                   help="where to download to (default downloads/<name>)")
    p.add_argument("--filter", default=None,
                   help="only files whose path contains this substring")
    p.add_argument("--compact", action="store_true",
                   help="also run compact_db.py when CBRS data was ingested")
    p.add_argument("--ask", action="store_true",
                   help="show the plan and wait for confirmation before running")
    p.epilog = ("any other flag is passed straight to ingest/fetch.py, "
                "e.g. --nist-only, --allow-host HOST, --allow-http")
    p.set_defaults(fn=cmd_get)

    for q in sub.choices.values():
        q.add_argument("--dry-run", action="store_true",
                       help="print the plan and change nothing")

    args, extra = ap.parse_known_args()

    # One preflight for every command. doctor is exempt: diagnosing a broken
    # environment is its whole job, and it reports the same facts in context.
    if getattr(args, "cmd", None) not in (None, "doctor"):
        bad = require_deps()
        if bad:
            print(bad, file=sys.stderr)
            return 1
        bad = db_dir_problem()
        if bad and args.cmd in ("get", "setup", "serve"):
            print(bad, file=sys.stderr)
            return 1

    if not getattr(args, "cmd", None):
        # No subcommand: work out the situation and offer the fix. This is the
        # path a first-time user should ever need.
        if extra:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
        return cmd_auto(ns())
    # Unknown flags are forwarded to fetch.py (--nist-only, --allow-host, ...);
    # anywhere else they are a typo and should not be swallowed.
    if extra and args.cmd != "get":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    args.fetch_args = extra
    return args.fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        # `atlas.py status | head` closes the pipe under us. That is the
        # reader's choice, not an error, but line buffering makes it land
        # mid-print and Python would otherwise report it as a crash. Redirect
        # the remaining flush to devnull so the interpreter cannot complain
        # about it during shutdown either.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:                                   # noqa: BLE001
            pass
        sys.exit(0)
