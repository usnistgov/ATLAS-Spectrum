"""
cbrs_files.py: finding and reading CBRS export CSVs.

psd_ingest.py and pfp_ingest.py differ only in the filename pattern they look
for and the shape of the numbers inside. Walking a directory, grouping matches
by sensor and day, explaining an empty result, and turning one CSV into
uint8-quantized rows are the same job in both, so they live here once. Each
caller passes its own compiled pattern and its own quantization range.

A matching filename must yield named groups `day`, `sensor` and `stat`.
"""

import os
import sys

import numpy as np


def walk_ext(root, exts):
    """Every file under root whose name ends in one of exts, sorted.

    A file path is returned as itself, so callers can accept either a folder
    or a single file without special-casing it.
    """
    if os.path.isfile(root):
        return [root] if root.lower().endswith(exts) else []
    out = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(exts):
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def discover(root, name_re, stat=None, collisions=None):
    """Walk root -> {sensor: {day: path}} for matching CSVs, any layout.

    The folder structure is deliberately ignored: the same call works on a
    download that preserved a record's directories, on a flat dump, and on the
    original Box share layout.

    Two files can claim the same sensor and day -- a re-export left beside the
    original, or the same day present in two subdirectories. Only one can be
    ingested, because the resume check keys on (sensor, day). This used to be
    decided by `os.walk` order and reported nowhere, so which of two differing
    exports you got depended on directory iteration order. Now the sorted-first
    path wins, deterministically, and every collision is recorded in
    `collisions` ({(sensor, day): [paths]}) for the caller to print. A run that
    silently picks between two versions of the same measurement is not a run you
    can reproduce.
    """
    found, seen = {}, {}
    for dirpath, _dirs, names in sorted(os.walk(root)):
        for n in sorted(names):
            m = name_re.match(n)
            if not m or (stat and m.group("stat") != stat):
                continue
            key = (m.group("sensor"), m.group("day"))
            path = os.path.join(dirpath, n)
            if key in seen:
                if collisions is not None:
                    collisions.setdefault(key, [seen[key]]).append(path)
                continue
            seen[key] = path
            found.setdefault(key[0], {})[key[1]] = path
    return found


def report_collisions(collisions, label="export"):
    """Print any (sensor, day) claimed by more than one file. -> how many."""
    if not collisions:
        return 0
    print(f"  WARN {len(collisions)} {label}(s) are claimed by more than one "
          f"file; the first path listed is the one ingested:", file=sys.stderr)
    for (sensor, day), paths in sorted(collisions.items()):
        print(f"    {sensor} {day}:", file=sys.stderr)
        for p in paths:
            print(f"      {p}", file=sys.stderr)
    print("    Remove or rename the duplicates so the run is reproducible.",
          file=sys.stderr)
    return len(collisions)


def stats_present(root, name_re):
    """Which statistics exist on disk, e.g. ['max', 'mean']."""
    out = set()
    for _, _, names in os.walk(root):
        for n in names:
            m = name_re.match(n)
            if m:
                out.add(m.group("stat"))
    return sorted(out)


def sample_of(root, limit=8):
    """A few real filenames under root, to show what the layout actually is."""
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names)[:limit]:
            out.append(os.path.relpath(os.path.join(dirpath, n), root))
            if len(out) >= limit:
                return out
    return out


def first_capture_time(path):
    """Epoch seconds of an export's FIRST capture, or None if unreadable.

    Reads two lines, not the file: the resume check runs once per candidate day
    and a real export is 12 MB. numpy parses the ISO stamp, which covers the
    real exports' "2025-07-09 00:02:39.258000+00:00" and the "...T00:00:00"
    form the fixtures use; anything else returns None and the caller falls back
    to the older, coarser test rather than guessing.
    """
    try:
        with open(path, "r", errors="replace") as f:
            f.readline()                       # header
            row = f.readline()
    except OSError:
        return None
    stamp = row.split(",", 1)[0].strip()
    if not stamp:
        return None
    # numpy wants a 'T' and no space before the offset, and treats a bare stamp
    # as UTC -- which is what these exports are.
    s = stamp.replace(" ", "T").rstrip("Z")
    # An explicit offset has to be SUBTRACTED, not discarded. numpy cannot
    # represent one, so it used to be stripped and the digits read as UTC -- but
    # "00:02:39+05:30" names an instant 5.5 hours earlier than "00:02:39Z", and
    # read_quantized (through DuckDB's TIMESTAMPTZ) stores the real instant. The
    # two then disagree by exactly the offset, so the resume check looked for a
    # capture that was never stored, decided the day was missing, and appended it
    # a second time: 6 rows where 3 belong, exit code 0, no warning. Invisible on
    # the max layer and quietly double-weighting every mean and median window.
    offset = 0.0
    tail = s[10:]
    for sign in ("+", "-"):
        if sign in tail:
            s = s[:10] + tail.split(sign)[0]
            hh, _, mm = tail.split(sign)[1].partition(":")
            try:
                offset = (int(hh) * 3600 + int(mm or 0) * 60) * \
                    (1.0 if sign == "+" else -1.0)
            except ValueError:
                return None
            break
    try:
        return float(np.datetime64(s, "us").astype("int64")) / 1e6 - offset
    except Exception:                                      # noqa: BLE001
        return None


def no_data(root, stat, name_re, pattern, label):
    """Explain an empty discovery instead of silently ingesting nothing.

    Returns 1 so callers can `sys.exit(no_data(...))`.
    """
    print(f"No {label} CSVs found under {os.path.abspath(root)}", file=sys.stderr)
    print(f"  looked for: {pattern}"
          + (f" with stat '{stat}'" if stat else "") + " (searched recursively)",
          file=sys.stderr)
    other = stats_present(root, name_re)
    if stat and other:
        print(f"  files with that name shape exist, but their stats are: "
              f"{', '.join(other)}. Try --stat {other[0]}.", file=sys.stderr)
    else:
        found = sample_of(root)
        if found:
            print("  what is actually there:", file=sys.stderr)
            for f in found:
                print(f"    {f}", file=sys.stderr)
        else:
            print("  that directory is empty.", file=sys.stderr)
    print("  Point --root at your copy of the data, or set SEA_DATA_ROOT "
          "(PowerShell: $env:SEA_DATA_ROOT=\"...\").", file=sys.stderr)
    return 1


def open_db(path, duckdb):
    """Connect for writing, or explain why not instead of a DuckDB traceback.

    The common failures are all environmental -- a directory that does not exist,
    an ATLAS_DB_DIR pointing at a file, a checkout someone else owns, or another
    writer holding the file -- and each used to surface as a bare IOException
    with a stack trace. build_psd_levels.py already does this; the ingests are
    what a new user runs first. `duckdb` is passed in so this module stays
    import-light.
    """
    parent = os.path.dirname(path) or "."
    if not os.path.isdir(parent):
        sys.exit(f"cannot write the database: {parent} is not a directory.\n"
                 f"  Create it, or point ATLAS_DB_DIR at somewhere that exists.")
    try:
        return duckdb.connect(path)
    except Exception as e:                                 # noqa: BLE001
        msg = str(e)
        hint = ("Another process holds it -- stop serve.py or the other ingest "
                "and re-run. DuckDB allows one writer at a time."
                if "lock" in msg.lower() else
                "Check the path exists and is writable, or set ATLAS_DB_DIR.")
        sys.exit(f"cannot open {path}\n  {msg}\n  {hint}")


def missing_root(root):
    """The message for a --root that does not exist at all."""
    return (f"source directory not found: {os.path.abspath(root)}\n"
            "  Pass --root /path/to/your/data, or set SEA_DATA_ROOT "
            "(PowerShell: $env:SEA_DATA_ROOT=\"...\").\n"
            "  To download it: python ingest/fetch.py <record-id> "
            "--dest <that directory>")


def resolve_sensor(sensor, found, root):
    """-> (sensor, None) or (None, message). Picks the only sensor when there
    is exactly one, and lists the real names when the guess was wrong."""
    if sensor is None:
        if len(found) == 1:
            return next(iter(found)), None
        return None, (f"which sensor? {len(found)} found under "
                      f"{os.path.abspath(root)}:\n  "
                      + "\n  ".join(sorted(found))
                      + "\n  Pass one as the first argument.")
    if sensor in found:
        return sensor, None
    near = [s for s in found if s.lower() == sensor.lower()]
    hint = (f"\n  Did you mean: {near[0]}" if near else
            "\n  Available: " + ", ".join(sorted(found)))
    return None, (f"no CSVs for sensor '{sensor}' under "
                  f"{os.path.abspath(root)}." + hint)


def read_quantized(path, csv_connection, nvals, qmin, qmax, unit, what, lead=()):
    """One export CSV -> (epoch_seconds[n], {lead column: values}, uint8[n, nvals]).

    Both deep layers store the same thing in the same way: a timestamp, zero or
    more key columns, then a fixed number of power values per capture, quantized
    to uint8 over a per-layer range. PSD has 2250 bins and no key column; PFP has
    560 frame positions keyed by `frequency`. Only those three numbers differ, so
    the reading, the column-count check, the quantization and the out-of-range
    warning are written once here.

    `lead` names the columns between the timestamp and the values, in order.
    """
    res = csv_connection.execute("SELECT * FROM read_csv_auto(?, header=true)", [path])
    cols = [c[0] for c in res.description]
    nlead = 1 + len(lead)
    if len(cols) - nlead != nvals:
        after = " and ".join(["timestamp", *lead]) if lead else "the timestamp"
        raise ValueError(f"expected {nvals} {what} columns after {after}, "
                         f"found {len(cols) - nlead}. This file is not a "
                         f"{nvals}-{'bin' if not lead else 'position'} CBRS "
                         f"{'PSD' if not lead else 'PFP'} export.")
    data = res.fetchnumpy()
    # A NULL cell arrives as a masked array, and np.asarray drops the mask and
    # hands back the fill DuckDB left underneath -- 0.0. Quantized, 0.0 dBm/Hz is
    # far above the ceiling, so a BLANK cell used to store 255: the loudest value
    # on the colour scale, an invented full-scale emitter where the measurement
    # was missing, against a real noise floor around -152. It then propagated
    # into the max pyramid and survived every zoom-out, and no warning fired
    # because a handful of cells is well under the old 1% threshold. So: fill
    # through the mask to NaN, and refuse the file. A blank has no correct byte,
    # and refusing one file is recoverable -- storing a fabricated peak is not.
    tcol = np.ma.filled(np.ma.asarray(data[cols[0]]),
                        np.datetime64("NaT")).astype("datetime64[us]")
    if bool(np.isnat(tcol).any()):
        raise ValueError(f"{int(np.isnat(tcol).sum())} row(s) have a blank or "
                         f"unparseable timestamp. A capture with no instant "
                         f"cannot be placed on the time axis (it used to be "
                         f"stored as NaN, which made psd_meta's range NaN and "
                         f"the viewer's JSON unparseable).")
    timestamps = tcol.astype("int64") / 1e6                   # epoch seconds
    keys = {name: _floats(data[cols[1 + i]])
            for i, name in enumerate(lead)}
    powers = np.stack([_floats(data[c]) for c in cols[nlead:]], axis=1)
    nf = int(np.count_nonzero(~np.isfinite(powers)))
    if nf:
        raise ValueError(f"{nf:,} of {powers.size:,} value cells are blank, NULL, "
                         f"NaN or infinite. There is no {unit} number to store "
                         f"for them; fix or drop those rows rather than have "
                         f"them rendered as real measurements.")
    quantized = np.clip(np.round((powers - qmin) / (qmax - qmin) * 255.0),
                        0, 255).astype(np.uint8)
    # Values outside the quantization range clip to a flat 0 or 255 instead of
    # failing, so a file in the wrong units (dBm rather than dBm/Hz, say)
    # ingests as a featureless band and looks like a rendering bug later. Name
    # the file now, while it is still obvious which one it was. Reported for ANY
    # clipped cell, not only past 1% of the file: the real exports clip a handful
    # of their loudest samples against the -90 ceiling, and eight silently
    # flattened peaks are exactly the thing worth knowing about.
    out = int(np.count_nonzero(~((powers >= qmin) & (powers <= qmax))))
    if out:
        print(f"  WARN {os.path.basename(path)}: {out:,} of {powers.size:,} "
              f"values ({100.0 * out / powers.size:.3g}%) are outside "
              f"[{qmin}, {qmax}] {unit} and were clipped flat")
    return timestamps, keys, quantized


def _floats(column):
    """One DuckDB column as float64, with NULLs preserved as NaN."""
    return np.ma.filled(np.ma.asarray(column).astype(np.float64), np.nan)
