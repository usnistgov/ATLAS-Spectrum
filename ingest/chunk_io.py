"""chunk_io.py: append to a database in whichever schema is already on disk.

The PSD and PFP databases exist in two shapes. The ingests write one row per
capture (`psd`, `pfp`); `compact_db.py` rewrites that into zlib-compressed
chunks of consecutive captures (`psd_chunk`, `pfp_chunk`) and serve.py reads
either one.

The catch is what happens *after* compaction. The CBRS datasets grow, so the
normal thing is to run the ingest again next month against a database that has
already been compacted. Writing rows into it then produces a file holding both
schemas at once, and serve.py resolves that by preferring the chunk table -- so
the freshly ingested month is stored, reported as ingested, and never drawn.
The old days are worse: the resume check reads the rows table, finds it empty,
and re-reads every day the chunk table already had, duplicating the lot.

So an append has to speak whichever schema it finds:

    kind = schema_of(connection, "psd")            # 'chunk' | 'rows' | 'empty'
    have = existing_days(connection, "psd", sensor, kind)
    with ChunkAppender(connection, "psd", sensor) as app:   # kind == 'chunk'
        app.add(times, quantized)

`existing_days` returns UTC date strings from whichever table holds the data,
which is what makes the resume comparison mean the same thing in both shapes.
"""

import zlib

import numpy as np

Z = 9                # zlib level, matching compact_db.py -- max level costs
                     # nothing at read time (decompress speed is level-independent)
                     # and measured ~2.6% smaller than level 6 on real PFP chunks
PSD_CHUNK = 256      # spectra per chunk
PFP_CHUNK = 1024     # frames per chunk

# (chunk table, row table, rows per chunk, extra key columns)
LAYOUT = {
    "psd": ("psd_chunk", "psd", PSD_CHUNK, ()),
    "pfp": ("pfp_chunk", "pfp", PFP_CHUNK, ("freq",)),
}


def tables(connection):
    """Every table name in this database.

    One definition rather than four: compact_db.py, build_psd_levels.py,
    recompress_chunks.py and serve.py each had their own copy of this query, and
    the point of `schema_of` below is that "what shape is this file in" has a
    single answer.
    """
    return {r[0] for r in connection.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}


def rollback_day(connection, appenders):
    """Undo the day in progress: the transaction AND any buffered captures.

    `appenders` is one ChunkAppender, an iterable of them, or None. Both ingests
    need exactly this, and getting only half of it right is a data bug: captures
    left in a buffer are written out with the NEXT day's flush, inside the next
    day's transaction, so a rolled-back day comes back to life attached to a day
    that succeeded.
    """
    if appenders is not None:
        if isinstance(appenders, ChunkAppender):
            appenders = [appenders]
        elif isinstance(appenders, dict):
            appenders = appenders.values()
        for app in appenders:
            app.reset()
    try:
        connection.execute("ROLLBACK")
    except Exception:                                      # noqa: BLE001
        pass                       # no transaction open; nothing to undo


def schema_of(connection, base):
    """Which shape this database is in: 'chunk', 'rows', or 'empty'.

    Mirrors serve.py's own detection, including its preference for the chunk
    table when a file somehow holds both -- an ingest must write where the
    server will read, not where it would rather write.
    """
    chunk, rows, _n, _k = LAYOUT[base]
    have = {r[0] for r in connection.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if chunk in have and connection.execute(
            f"SELECT count(*) FROM {chunk}").fetchone()[0]:
        return "chunk"
    if rows in have and connection.execute(
            f"SELECT count(*) FROM {rows}").fetchone()[0]:
        return "rows"
    return "empty"


def stored_times(connection, base, sensor, kind):
    """Every capture instant stored for this sensor, as epoch seconds.

    One read for the whole sensor, so callers can test many candidate files
    against it without a query each.
    """
    chunk, rows, _n, _k = LAYOUT[base]
    if kind == "rows":
        return np.array([r[0] for r in connection.execute(
            f"SELECT t FROM {rows} WHERE sensor=?", [sensor]).fetchall()],
            dtype=np.float64)
    if kind != "chunk":
        return np.empty(0, np.float64)
    parts = []
    for (blob,) in connection.execute(
            f"SELECT times FROM {chunk} WHERE sensor=?", [sensor]).fetchall():
        ts = np.frombuffer(zlib.decompress(blob), dtype=np.float64)
        if ts.size:
            parts.append(ts)
    return np.concatenate(parts) if parts else np.empty(0, np.float64)


def captures_per_day(times):
    """{'YYYY-MM-DD': how many captures are stored on that UTC day}."""
    if not len(times):
        return {}
    days, counts = np.unique((np.asarray(times) // 86400).astype(np.int64),
                             return_counts=True)
    return {str(np.datetime64(int(d), "D")): int(n)
            for d, n in zip(days, counts)}


# A day that a previous file only STRADDLED into holds a capture or two just
# after midnight; a day that was really ingested holds hundreds (the real exports
# carry 700-800). So the count alone decides almost every case, and the file only
# has to be opened for the handful in between.
STRADDLE_MAX = 4


def day_is_ingested(day, per_day, stored_ms, first_time):
    """Is this day's export already stored? `first_time` is called only if needed.

    Reading each candidate file's own first capture is the exact test (see
    already_have), but "exact" was costing far too much: on a network or
    on-demand filesystem -- Box Drive, OneDrive, a mounted share -- opening a
    file is a download, and this ran for every one of ~9,500 exports BEFORE the
    first capture was ingested. Measured: 40 minutes with zero bytes written.

    The count of stored captures on that day settles it without opening
    anything:

      0 captures            -> not ingested. Nothing to read.
      more than STRADDLE_MAX -> ingested for real; a straddle cannot make
                               hundreds of captures. Nothing to read.
      1..STRADDLE_MAX       -> genuinely ambiguous, and the only case that
                               reads the file.

    So a fresh database opens nothing, a normal resume opens nothing, and the
    exact check still runs exactly where the straddle bug lived.
    """
    n = per_day.get(day, 0)
    if n == 0:
        return False
    if n > STRADDLE_MAX:
        return True
    t = first_time()
    if t is None:
        return True          # cannot tell; treat as done rather than duplicate
    return already_have(stored_ms, t)


def already_have(stored_ms, t):
    """Is this exact capture instant already stored? (milliseconds, as a set.)

    The resume test this replaces asked "does the database hold any capture on
    the day this file is NAMED for", and that is not the same question. A CBRS
    export runs from just after midnight to just after the NEXT midnight -- the
    real ones end at 00:00:22 on day D+1 -- so ingesting day D always put a
    capture into day D+1 and marked D+1 complete. Day D+1's file was then
    skipped in full, silently: measured on three real days, 2,349 captures
    became 1,584, a third of the data gone, exit code 0.

    A file's OWN first capture is an exact fingerprint instead: it is present
    only if that file was actually read. Compared in integer milliseconds so the
    float64 round trip cannot make an equal instant look different.
    """
    return int(round(t * 1000)) in stored_ms


def existing_days(connection, base, sensor, kind):
    """UTC dates ('YYYY-MM-DD') this sensor already has stored, either shape.

    Bucketed in UTC on both sides. `to_timestamp()` renders in the machine's
    local zone, so west of Greenwich an 00:30 UTC capture would bucket to the
    previous date and its day would be re-read on every run.
    """
    chunk, rows, _n, _k = LAYOUT[base]
    if kind == "rows":
        return {str(r[0]) for r in connection.execute(
            "SELECT DISTINCT CAST(to_timestamp(t) AT TIME ZONE 'UTC' AS DATE) "
            f"FROM {rows} WHERE sensor=?", [sensor]).fetchall()}
    if kind != "chunk":
        return set()
    # A chunk stores its capture instants as a compressed float64 blob. Its
    # t0..t1 span cannot be used instead: a chunk that begins late on one day
    # and ends early on the next would mark both days complete when neither is,
    # and the missing captures would never be picked up. So read the instants.
    days = set()
    for (blob,) in connection.execute(
            f"SELECT times FROM {chunk} WHERE sensor=?", [sensor]).fetchall():
        ts = np.frombuffer(zlib.decompress(blob), dtype=np.float64)
        if ts.size:
            days.update(np.unique(
                (ts // 86400).astype(np.int64)).tolist())
    return {str(np.datetime64(int(d), "D")) for d in days}


def _sql(base, extra):
    chunk, _r, _n, keys = LAYOUT[base]
    cols = ["sensor", *keys, "t0", "t1", "n", "times"] + list(extra)
    return (f"INSERT INTO {chunk} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})")


class ChunkAppender:
    """Buffer captures and flush them as whole chunks, as compact_db would.

    Appends only: chunks already in the table are never rewritten, so a resumed
    run costs nothing for the days it already has. The callers flush at the end
    of each DAY, inside that day's transaction, so a day that did not finish
    contributes nothing -- which leaves one short chunk per day rather than per
    run. Harmless to read, and a later `compact_db.py` pass repacks them.
    """

    def __init__(self, connection, base, sensor, payload_col, key=None):
        self.connection, self.base, self.sensor = connection, base, sensor
        self.key = () if key is None else (key,)
        self.limit = LAYOUT[base][2]
        self.sql = _sql(base, (payload_col,))
        self.ts, self.buf, self.total = [], [], 0

    def add(self, times, payloads):
        """`times` epoch seconds, `payloads` uint8 arrays, one per capture."""
        for t, p in zip(times, payloads):
            self.ts.append(float(t))
            self.buf.append(p.tobytes() if hasattr(p, "tobytes") else p)
            if len(self.buf) >= self.limit:
                self.flush()

    def reset(self):
        """Throw away whatever is buffered but not yet written.

        Used on the rollback path: a day that failed must contribute nothing, and
        captures still sitting in this buffer would otherwise be written out with
        the NEXT day's flush -- inside the next day's transaction, so a rolled
        back day would come back to life attached to a day that succeeded.
        """
        self.ts, self.buf = [], []

    def flush(self):
        if not self.buf:
            return
        ts = np.array(self.ts, dtype=np.float64)
        self.connection.execute(self.sql, [
            self.sensor, *self.key, float(ts[0]), float(ts[-1]), len(self.buf),
            zlib.compress(ts.tobytes(), Z),
            zlib.compress(b"".join(self.buf), Z)])
        self.total += len(self.buf)
        self.ts, self.buf = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.flush()
        return False


def stored_span(connection, base, sensor, kind):
    """(t_min, t_max, captures) for this sensor, whichever shape is on disk."""
    chunk, rows, _n, _k = LAYOUT[base]
    if kind == "chunk":
        r = connection.execute(f"SELECT min(t0), max(t1), coalesce(sum(n), 0) "
                        f"FROM {chunk} WHERE sensor=?", [sensor]).fetchone()
    else:
        r = connection.execute(f"SELECT min(t), max(t), count(*) "
                        f"FROM {rows} WHERE sensor=?", [sensor]).fetchone()
    return r[0], r[1], r[2] or 0
