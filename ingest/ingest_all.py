"""ingest_all.py: point it at a folder, get every layer the viewer can draw.

    python ingest/ingest_all.py <folder>

One pass over any folder of exports: IQ captures, CBRS PSD (max, median and
mean), CBRS PFP, and Summaries CSVs, in whatever mix is there and at whatever
depth, followed by compaction and the coarse PSD pyramid. A file is recognised
by its extension, its filename shape, or its own CSV header -- the rule atlas.py
already uses, imported rather than restated -- so a folder called "spectra" full
of PFP files is read as PFP. Two layout caveats that are real: directories whose
names are in atlas.SKIP_DIRS (tmp, env, node_modules and friends) and any
dot-directory are not searched at all, so exports parked in one are invisible.

    python ingest/ingest_all.py <folder> --dry-run      # show the plan, do nothing
    python ingest/ingest_all.py <folder> --dest DIR     # write databases elsewhere
    python ingest/ingest_all.py <folder> --no-compact   # skip compaction
    python ingest/ingest_all.py <folder> --no-levels    # skip the coarse pyramid

Safe to re-run, which is the normal case: next month's files arrive and you run
the same command. Every ingest skips days it already holds, compaction skips
databases already in the compact shape, and the pyramid tops up from where it
stopped. A file that cannot be ingested is reported with the reason and the run
carries on, so one bad export never costs you the other nine.

WHAT "ANY DATA" HONESTLY MEANS, in three parts.

Bin count IS checked. The viewer's PSD path is built around 2250 bins and PFP
around 560 frame positions; those numbers live in serve.py and in the stored
schema, not just here. Preflight counts the columns in a SAMPLE of each
(sensor, stat) group -- PREFLIGHT_SAMPLE files, or all of them with
ATLAS_PREFLIGHT_ALL=1 -- and reports a mismatch before anything runs, so you
learn it from the plan instead of halfway through a long job. It reads one header
line with Python's csv module rather than opening the file with DuckDB the way
the ingest does: on an on-demand filesystem every open is a download, and doing
it the thorough way cost 40 minutes before a single capture was stored. Note the limit of that:
preflight is a REPORT, not a gate -- the per-layer ingest scripts scan their own
--root and will still meet the file and reject it themselves. What preflight
buys you is the reason, up front, per file.

The frequency axis is NOT checked, and cannot be. psd_ingest writes the band
start and bin spacing from its own constants (3530.04 MHz, 80 kHz), because a
real export's header is just b0..b2249 and carries no frequency information. A
2250-bin export from a different band or spacing therefore ingests and is
silently labelled with the CBRS axis. If your data is not that band, the pictures
will be correct in shape and wrong in frequency, and nothing here can tell.
Values far outside the layer's quantization range are the one hint, and that
warning from the ingest is surfaced in the summary rather than swallowed.

Statistics are filtered on purpose. PSD keeps max, median and mean, each in its
own sibling database, because those are the three serve.py reads; psd_ingest
would happily write psd_p99.duckdb and nothing would ever open it. PFP is
different and worse: there is ONE pfp table with no stat column, so ingesting two
PFP statistics interleaves them into a single layer whose metadata names only one
-- measured as a 50 dB step mid-picture. Exactly one PFP statistic is ingested
and the others are named in the report.
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atlas                                             # noqa: E402
import pfp_ingest                                        # noqa: E402
import psd_ingest                                        # noqa: E402

# The statistics the viewer can actually display. psd_ingest accepts any word
# and would write psd_<word>.duckdb; serve.py only ever reads these three, so
# anything else is named in the report instead of built.
PSD_STATS = ("max", "median", "mean")

# How many exports per sensor-and-statistic the geometry preflight opens. Two is
# enough to catch a dataset in the wrong shape; ATLAS_PREFLIGHT_ALL=1 checks every
# file, which is thorough and, on an on-demand filesystem, very slow.
PREFLIGHT_SAMPLE = 2
PREFLIGHT_ALL = os.environ.get("ATLAS_PREFLIGHT_ALL") == "1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


UNREADABLE_RE = re.compile(r"(\d+) file\(s\) failed to read")
# Warnings the per-layer scripts print that a wrapper must not swallow. Two of
# them are the only signal for whole classes of silent wrongness: PFP and IQ
# report unread files and then exit 0, and the clipping warning is what tells you
# an export is in the wrong units or band (the frequency axis itself cannot be
# checked -- see the module docstring).
NOTABLE = (re.compile(r"^\s*WARN .*", re.M),
           re.compile(r"^.*file\(s\) failed to read.*$", re.M),
           re.compile(r"^.*capture\(s\) failed to read.*$", re.M))


def notable_lines(text):
    seen, out = set(), []
    for rx in NOTABLE:
        for m in rx.finditer(text):
            line = m.group(0).strip()
            if line and line not in seen:
                seen.add(line); out.append(line)
    return out


def run(argv, label, env=None, expected_bad=0):
    """-> (state, output) where state is "ok" | "partial" | "failed".

    Never raises: a failing step is data, not an exception.

    `expected_bad` is how many files in this step's folder preflight already
    ruled out. The per-layer ingests scan their own --root, so they meet those
    files again and exit non-zero for them -- correctly, on their own terms, but
    it made a run that ingested every usable file report itself as FAILED. When
    the count of unreadable files matches what preflight predicted AND the step
    still ingested something, that is a partial with a cause already printed,
    not a failure. A HIGHER count means something new went wrong and is still a
    failure, which is the distinction worth drawing.
    """
    # Streamed, not captured. capture_output=True meant a step that runs for an
    # hour printed absolutely nothing until it finished, so there was no way to
    # tell progress from a hang -- and the summaries step, reading gigabytes off a
    # network drive, is exactly that step. The output is still collected for the
    # notable-line scan below; it is just also echoed as it arrives.
    try:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=1)
    except OSError as e:                                  # noqa: BLE001
        return "failed", str(e)
    lines = []
    for line in p.stdout:
        lines.append(line)
        print(f"    | {line.rstrip()}", flush=True)
    p.wait()
    both = "".join(lines)
    for line in notable_lines(both):
        log(f"  ! {line}")
    if p.returncode == 0:
        # pfp_ingest and iq_ingest exit 0 even when some files could not be read,
        # so a clean exit code is not the same as a clean run. Say so.
        if any("failed to read" in ln for ln in notable_lines(both)):
            return "partial", both
        return "ok", both
    m = UNREADABLE_RE.search(both)
    if m and expected_bad and int(m.group(1)) <= expected_bad and "Done in" in both:
        log(f"  partial: {label} -- ingested everything usable; "
            f"{m.group(1)} file(s) named above cannot be read")
        return "partial", both
    log(f"  FAILED {label} (exit {p.returncode})")
    for line in both.strip().splitlines()[-3:]:
        log(f"    {line}")
    return "failed", both


# ---- what is in the folder -------------------------------------------------

def survey(root, max_files, max_depth):
    """-> {kind: {directory: [filenames]}} plus the leftovers.

    Directories matter only because the per-layer ingest scripts take a --root
    and scan it themselves; the CLASSIFICATION is atlas.kind_of, so a folder
    named "spectra" holding PFP files is read as PFP.
    """
    out = {k: {} for k in ("iq", "psd", "pfp", "summaries", "duckdb")}
    unknown = []
    gen, limits = atlas.walk_limited(root, max_files=max_files, max_depth=max_depth)
    for dirpath, name in gen:
        kind = atlas.kind_of(dirpath, name)
        if kind:
            out[kind].setdefault(dirpath, []).append(name)
        elif not name.lower().endswith(atlas.COMPANION_EXT):
            unknown.append(os.path.join(dirpath, name))
    return out, unknown, limits


_CSV_CON = None


def csv_width(path):
    """-> (column count, None) or (None, why it cannot be read).

    Reads ONE line and counts fields with the csv module, which handles a quoted
    field containing a comma the same way the reader does. Splitting on "," got
    that wrong (it reported a good file as refused); handing every file to
    DuckDB's read_csv_auto got it right but cost ~1.8 s per 2251-column export
    even on a local disk, because the sniffer samples rows to infer types. The
    header is all this needs.

    DuckDB is still the arbiter for the one case a comma count cannot judge: a
    file whose delimiter is not a comma looks like a single column, and DuckDB
    sniffs the real delimiter. That is rare, so paying for it there is fine.
    """
    global _CSV_CON
    import csv
    try:
        with open(path, "r", errors="replace", newline="") as f:
            head = f.readline()
    except OSError as e:
        return None, str(e)
    if not head.strip():
        return None, "file is empty"
    n = len(next(csv.reader([head])))
    if n > 1:
        return n, None
    import duckdb
    if _CSV_CON is None:
        _CSV_CON = duckdb.connect()
    try:
        res = _CSV_CON.execute(
            "SELECT * FROM read_csv_auto(?, header=true) LIMIT 0", [path])
        return len(res.description), None
    except Exception as e:                                 # noqa: BLE001
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return None, f"unreadable as CSV: {first[:120]}"


def preflight(kind, directory, names):
    """-> (usable filenames, [(filename, why not)]).

    The geometry the viewer is built around is checked BEFORE anything is
    ingested, because a 1024-bin export does not become a 2250-bin one halfway
    through and the failure is far more legible here than inside a numpy stack.
    """
    want = psd_ingest.NF if kind == "psd" else pfp_ingest.NPOS
    lead = 1 if kind == "psd" else 2          # timestamp [+ frequency for PFP]
    ok, bad = [], []
    # SAMPLED, not exhaustive. Bin count is a property of the export product, not
    # of one file, so reading a couple per group answers "is this dataset the
    # shape the viewer draws" -- which is the actual question. Reading every file
    # answered it ~9,500 times, and on a network or on-demand filesystem (Box
    # Drive, OneDrive, a mounted share) each open is a download: measured 40
    # minutes before the first capture was ingested. The ingest itself still
    # rejects any individual bad file when it reaches it, with the same message.
    todo = sorted(names)
    checked = todo if PREFLIGHT_ALL else todo[:PREFLIGHT_SAMPLE]
    for n in checked:
        cols, err = csv_width(os.path.join(directory, n))
        if cols is None:
            bad.append((n, err)); continue
        got = cols - lead
        if got != want:
            bad.append((n, f"{got} value columns, the viewer's {kind.upper()} path "
                           f"needs exactly {want}"))
        else:
            ok.append(n)
    # Files not sampled are assumed usable: the ingest is the authority, this is
    # a heads-up. Without this they would look "unusable" and the group would be
    # skipped entirely.
    ok += [n for n in todo if n not in set(checked)]
    return ok, bad


def by_sensor_stat(kind, names):
    """Group filenames by (sensor, stat), the unit each ingest step covers.

    Preflight used to be per DIRECTORY while the steps are per sensor and
    statistic, so one sensor's unusable files licensed another sensor's real
    failures -- a genuinely broken day was reported as a geometry refusal and
    the run still exited 0. Grouping first makes the count that classifies a
    step describe exactly that step.
    """
    regex = psd_ingest.NAME_RE if kind == "psd" else pfp_ingest.NAME_RE
    out = {}
    for n in names:
        m = regex.match(n)
        if m:
            out.setdefault((m.group("sensor"), m.group("stat")), []).append(n)
    return out


# ---- the plan --------------------------------------------------------------

def build_plan(found, dest_env, scan_root):
    """-> (steps, notes). A step is (label, argv, env, files preflight ruled out)."""
    steps, notes = [], []

    for d in sorted(found["iq"]):
        steps.append((f"IQ captures in {d}",
                      atlas.script("iq_ingest.py", d, "--dataset",
                                   os.path.basename(os.path.normpath(d))),
                      dest_env, 0))

    for d, names in sorted(found["psd"].items()):
        groups = by_sensor_stat("psd", names)
        extra = sorted({stat for (_s, stat) in groups if stat not in PSD_STATS})
        if extra:
            notes.append(f"{d}: ignoring statistic(s) {', '.join(extra)} -- the "
                         f"viewer reads only {', '.join(PSD_STATS)}")
        for (sensor, stat) in sorted(groups):
            if stat not in PSD_STATS:
                continue
            good, bad = preflight("psd", d, groups[(sensor, stat)])
            for n, why in bad:
                notes.append(f"skipped {n}: {why}")
            if not good:
                # Every file for this sensor and statistic is unusable, so there
                # is no step to run. Planning one anyway meant a permanent
                # failure and "re-run to retry" advice that could never work.
                notes.append(f"no usable {stat} files for sensor {sensor} in {d}")
                continue
            steps.append((f"PSD {stat}, sensor {sensor}, in {d}",
                          atlas.script("psd_ingest.py", sensor, "--stat", stat,
                                       "--root", d), dest_env, len(bad)))

    for d, names in sorted(found["pfp"].items()):
        groups = by_sensor_stat("pfp", names)
        stats = sorted({stat for (_s, stat) in groups})
        if not stats:
            continue
        # ONE statistic only. The pfp table has no stat column and there is no
        # per-stat sibling database, so a second statistic interleaves into the
        # same layer while pfp_meta names only one of them.
        chosen = "max_peak" if "max_peak" in stats else stats[0]
        if len(stats) > 1:
            notes.append(f"{d}: PFP has statistic(s) {', '.join(stats)}; ingesting "
                         f"'{chosen}' only -- the PFP layer stores one statistic "
                         f"per sensor and mixing them corrupts it")
        for (sensor, stat) in sorted(groups):
            if stat != chosen:
                continue
            good, bad = preflight("pfp", d, groups[(sensor, stat)])
            for n, why in bad:
                notes.append(f"skipped {n}: {why}")
            if not good:
                notes.append(f"no usable {stat} files for sensor {sensor} in {d}")
                continue
            steps.append((f"PFP {stat}, sensor {sensor}, in {d}",
                          atlas.script("pfp_ingest.py", sensor, "--stat", stat,
                                       "--root", d), dest_env, len(bad)))

    # ONE summaries step, pointed at the scan root. build_db.py REPLACES the
    # summary database rather than appending to it, so one step per directory
    # meant every directory but the last was thrown away -- two months of
    # summaries became one, while PSD and PFP correctly accumulated.
    #
    # The root is a cheap argument now, and only now: build_db.py classifies
    # candidates by FILENAME through atlas.kind_of, so the ~44,600 PSD and PFP
    # exports under a real dataset root cost nothing. Handing it the root while it
    # still opened every .csv to read its columns was what made this step appear
    # to hang for over an hour on a Box Drive folder -- every open is a download.
    if found["summaries"]:
        dirs = sorted(found["summaries"])
        if len(dirs) > 1:
            notes.append(f"Summaries CSVs in {len(dirs)} directories; building "
                         f"them together in one pass over {scan_root}")
        steps.append((f"Summaries under {scan_root}",
                      atlas.script("build_db.py", "--csv-dir", scan_root),
                      dest_env, 0))

    return steps, notes


def psd_stats_on_disk(db_dir):
    """Which sibling PSD databases exist, so levels are built for those only."""
    out = []
    for stat in PSD_STATS:
        name = "psd.duckdb" if stat == "max" else f"psd_{stat}.duckdb"
        if os.path.exists(os.path.join(db_dir, name)):
            out.append(stat)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="directory holding the exports (searched recursively)")
    ap.add_argument("--dest", default=None,
                    help="where the databases go (default: the ATLAS folder, "
                         "or ATLAS_DB_DIR)")
    ap.add_argument("--no-compact", action="store_true", help="skip compaction")
    ap.add_argument("--no-levels", action="store_true",
                    help="skip the coarse PSD pyramid (wide windows stay slow)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the preflight findings, ingest nothing")
    ap.add_argument("--max-files", type=int, default=200000)
    ap.add_argument("--max-depth", type=int, default=12)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")

    env = dict(os.environ)
    if args.dest:
        dest = os.path.abspath(os.path.expanduser(args.dest))
        os.makedirs(dest, exist_ok=True)
        # Both the umbrella variable and the per-database ones: the ingest
        # scripts read the specific variable first, so setting only ATLAS_DB_DIR
        # would be silently ignored by any of them that has its own.
        env["ATLAS_DB_DIR"] = dest
        for name, var in (("spectrum", "SPECTRUM_DB"), ("psd", "PSD_DB"),
                          ("pfp", "PFP_DB"), ("iq", "IQ_DB")):
            env[var] = os.path.join(dest, f"{name}.duckdb")
    # Resolved the way the ingest scripts themselves resolve it: the specific
    # variable wins over ATLAS_DB_DIR, and a relative path is made absolute
    # against the repo, because that is the cwd the children run in. Reading only
    # ATLAS_DB_DIR (and not absolutising it) meant compaction and every pyramid
    # step were silently skipped while the run reported total success and printed
    # a directory nothing had been written to.
    psd_db = env.get("PSD_DB")
    if psd_db:
        db_dir = os.path.dirname(os.path.abspath(os.path.join(ROOT, psd_db)))
    else:
        db_dir = os.path.abspath(os.path.join(ROOT, env.get("ATLAS_DB_DIR") or ROOT))

    log(f"scanning {root}")
    found, unknown, limits = survey(root, args.max_files, args.max_depth)
    counts = {k: sum(len(v) for v in found[k].values()) for k in found}
    log("found: " + ", ".join(f"{k}={counts[k]}" for k in
                              ("iq", "psd", "pfp", "summaries", "duckdb")))
    # `limits` is a dict, not a flag: {"files": bool, "depth": <folders skipped>}.
    # Treating it as truthy warned on every run and printed the raw dict. A scan
    # that quietly stopped early and then reported what it found is the one
    # failure mode worth being loud about, so report each budget on its own.
    if limits.get("files"):
        log(f"WARNING: stopped at the {args.max_files} file limit; some files "
            f"were NOT looked at. Re-run with a bigger --max-files.")
    if limits.get("depth"):
        log(f"WARNING: stopped at {args.max_depth} directory levels; "
            f"{limits['depth']} deeper folder(s) were NOT searched. "
            f"Re-run with a bigger --max-depth.")

    steps, notes = build_plan(found, env, root)
    for n in notes:
        log(f"note: {n}")
    if unknown:
        log(f"note: {len(unknown)} file(s) were not recognised as any layer, "
            f"e.g. {os.path.basename(unknown[0])}")
    if found["duckdb"]:
        n = sum(len(v) for v in found["duckdb"].values())
        log(f"note: {n} prebuilt .duckdb file(s) present; this tool builds from "
            f"exports and leaves them alone (use atlas.py get to adopt one)")

    if not steps:
        log("nothing to ingest. Expected IQ captures, or CBRS exports named "
            "<YYYY-MM-DD>_<sensor>_<stat>.csv / PFP_<YYYY-MM-DD>_<sensor>_<stat>.csv, "
            "or Summaries CSVs.")
        return 1

    log(f"{len(steps)} ingest step(s) planned")
    if args.dry_run:
        for label, argv, _env, _bad in steps:
            log(f"  would run: {label}")
        if not args.no_compact:
            log("  would run: compact_db")
        if not args.no_levels:
            log("  would run: build_psd_levels for each PSD statistic present")
        return 0

    results = {}
    for i, (label, argv, senv, bad) in enumerate(steps, 1):
        log(f"[{i}/{len(steps)}] {label}")
        results[label] = run(argv, label, senv, expected_bad=bad)[0]

    # Compaction before the pyramid: levels built over the row schema used to be
    # discarded by compaction. compact_db.py carries them across now, so the
    # order is no longer load-bearing -- it is just cheaper this way, since the
    # pyramid then reads the compact shape.
    if not args.no_compact:
        log("compacting")
        # compact_db.py exits 0 even when it REFUSED a swap or left a build file
        # incomplete, so the exit code alone would report those as success. Read
        # its own words for the two outcomes that matter.
        state, tail = run(atlas.script("compact_db.py"), "compact_db", env)
        refused = "REFUSING" in tail or "INCOMPLETE" in tail
        if refused:
            log("  ! compaction refused a swap or left a build file incomplete -- "
                "the live databases were NOT replaced (details above)")
        results["compact_db"] = "failed" if refused else state
    if not args.no_levels:
        for stat in psd_stats_on_disk(db_dir):
            log(f"building coarse PSD levels: {stat}")
            results[f"levels {stat}"] = run(
                atlas.script("build_psd_levels.py", "--stat", stat),
                f"levels {stat}", env)[0]

    failed = [k for k, v in results.items() if v == "failed"]
    partial = [k for k, v in results.items() if v == "partial"]
    print()
    log("=" * 60)
    log(f"{len(results) - len(failed)} of {len(results)} step(s) succeeded"
        + (f" ({len(partial)} with unreadable file(s) skipped)" if partial else ""))
    for k in failed:
        log(f"  FAILED: {k}")
    if failed:
        log("Re-run the same command to retry: every step skips work already done.")
    elif partial:
        log("The skipped files are listed above and will never ingest -- their "
            "geometry does not match what the viewer draws. Everything else is in.")
    log(f"databases in {db_dir}")
    log("start the viewer with: python serve.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
