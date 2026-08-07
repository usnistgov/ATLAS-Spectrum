# ATLAS: 30-combination data verification and adverse-condition testing

What this records: the August 2026 verification of the CBRS PSD databases against
the physical NASCTN exports, and the adverse-condition testing that went with it.
Kept because it is the evidence behind "the data is 1:1", it names the eight cells
where that is not literally true, and it lists the findings that were deliberately
NOT changed together with the reasoning — which is otherwise unrecoverable without
re-running the whole exercise.

Two questions were asked. Is the data 1:1 with the physical CSVs, for all ten
sensors in all three statistics? And does a clean install hold up under bad
conditions?

Short answers: **the data is right**, and **the pipeline was not durable** — five
separate ways a re-run could lose or corrupt data, all now fixed and covered by
`examples/test_durability.py`.

---

## Part 1 — the data, all 30 combinations

### What was checked

All ten sensors, all three statistics, against the physical NASCTN exports:

| | max | median | mean |
|---|---|---|---|
| GMM | ✅ | ✅ | ✅ |
| HU (3 consecutive days) | ✅ | ✅ | ✅ |
| Midway | ✅ | ✅ | ✅ |
| NIT | ✅ | ✅ | ✅ |
| Catalina-Directional | ✅ | ✅ | ✅ |
| Catalina-Omni | ✅ | ✅ | ✅ |
| CBBT-Directional | ✅ | ✅ | ✅ |
| CBBT-Omni | ✅ | ✅ | ✅ |
| PtLoma-Directional | ✅ | ✅ | ✅ |
| PtLoma-Omni | ✅ | ✅ | ✅ |

Each combination was built from scratch through the whole real path — ingest,
`compact_db.py`, `build_psd_levels.py` — and then compared against the CSV.
Not sampled: **every cell**. 36 export files, 46,068,742 values.

**Result: 30 of 30 verified 1:1.** Worst error across all 46 million values is
0.176471 dB, which is *exactly* half a quantization step — the arithmetic floor,
what a correct round-to-nearest cannot do better than. Zero values exceed it.

Note on PtLoma: those two sensors were not deployed on 2025-07-09, which is why
they needed a later date (2026-01-15). Both names in `ingest_all_stats.py` are
correct — I confirmed them against your own `spectrum.duckdb`.

### What "1:1" was tested to mean

Six independent things, because each catches a different kind of wrong:

- **Values.** Dequantize the stored byte, compare to the CSV float. This is the
  only formula-agnostic test — it does not care how the ingest quantizes, only
  that the byte means the CSV's number.
- **Bytes.** Recompute the expected byte from the CSV with exact rational
  arithmetic (`fractions.Fraction`, no floating point anywhere in the reference).
  This matters because at values landing exactly halfway between two bytes,
  *both* bytes are within half a step and the value check is structurally blind.
  786,287 such cells exist in this data — 1.7% — and all 786,287 were checked
  individually against the round-half-to-even rule. All correct.
- **Routing.** Each database must match its own statistic's CSV *and must not
  match the other two*. Without this, an ingest that filled the mean layer with
  max data would pass everything else. Measured separation: 16 to 45 dB.
- **Geometry.** Timestamps to the microsecond, capture order, bin order.
- **The frequency axis.** This was previously an open question — and it turns out
  the CSVs carry the frequency of every column in their header row. Verified: the
  headers match `3530.040 MHz + i × 80 kHz` for all 2250 bins with **max
  deviation 0.0 Hz**, first bin 3530040000, last 3709960000. Off-by-one or
  reversed alignment produces 13–67 dB errors, so the test discriminates by two
  orders of magnitude.
- **The coarse pyramid.** Every `psd_lvl` row recomputed from the CSV floats
  directly — not from the database's own fine table, which would pass on a
  pyramid that is merely self-consistent with wrong data.

### The one real 1:1 exception

**8 cells out of 46,068,742 clip against the quantization ceiling.** The store
covers [−180, −90] dBm/Hz; a handful of the loudest samples exceed −90 and are
pinned to exactly −90.0:

| file | cells | worst error |
|---|---|---|
| `2025-07-09_HU_max.csv` | 6 | 0.44 dB |
| `2025-07-10_HU_max.csv` | 1 | 0.70 dB |
| `2026-01-15_PtLoma-Directional_max.csv` | 1 | 0.20 dB |

All in the max statistic, all around 3701 MHz. The true peak of the HU max
dataset is −89.3 dBm/Hz and reads back as −90.0.

**I did not widen the range.** Doing so would mean re-ingesting all 22 GB, and
0.7 dB on eight samples is not worth that risk. What changed is that it is no
longer silent: the out-of-range warning used to fire only past 1% of a file
(these are 0.0001%), and now reports any clipped cell with its count.

### Independent confirmation

Five agents were given the same job and told explicitly *not* to use my verifier
— to write their own readers, their own CSV parsers, their own chunk decoders.
Independent totals: 25,983,000 values with 0 mismatches (GMM/HU/Midway),
10.4 M bytes bit-identical (Catalina/NIT), 5,231,250 bytes byte-identical across
the row→chunk conversion (PtLoma), and 36 of 36 tiles served through serve.py's
own read path byte-identical to the databases. A sixth agent's only job was to
break my verifier — and it did (see below).

### My verifier was defective, and that is now tested

The first version reported 30/30 while being unable to detect a round-half-up
quantizer, blessing 173 of 173 wrong pyramid rows, and never exercising which CSV
fed which database. All three are fixed, and `examples/test_verify_1to1.py` now
damages a real database every way the ingest could plausibly get one wrong —
including that exact rounding bug — and requires each to be caught *by a check
that owns it*, not incidentally by an unrelated one. The 30/30 above is from the
hardened version.

---

## Part 2 — bad conditions

### Environment hostility: 35 of 36 runs bit-identical

The reference database was rebuilt under each condition and the stored bytes
required to be **bit-identical**, not merely close.

Identical: `TZ` at +14, −9, +05:45, UTC, unset, and two malformed values; a
mixed-timezone resume across a midnight-straddling export; `LC_ALL` de_DE (comma
decimals), C, tr_TR (the dotless-i trap); paths with spaces, non-ASCII, 1013
characters, `#`, `$`; umask 077, non-root, `HOME` unset, read-only repo; Python
3.10 and 3.13; dependency floors as old as duckdb 1.0.0 / numpy 1.26.0. Rendered
WebP tiles were md5-identical across three timezones. Zero outbound network
connections from either the ingest or the server.

The timezone result is the one worth dwelling on: it holds even for offset-free
timestamps under `TZ=Pacific/Kiritimati`, which shows the UTC discipline does not
depend on the `+00:00` the real exports happen to carry.

The one failure was a path containing a single quote (`~/Sean's data`), which
aborted compaction with a SQL parser error. Fixed.

### Five ways a re-run could lose or corrupt data — all fixed

Every one of these was reproduced, not theorised, and every one **exited 0**
while the data was wrong. That is the only kind of failure that survives to
production.

**1. A day written part-way was recorded as complete.** DuckDB autocommits per
parameter set, so a SIGKILL, a Ctrl+C, an OOM kill or a full disk left an
arbitrary prefix of the day committed — and the resume check, which asks whether
a day was *started*, then skipped the rest forever. Measured: 2,349 captures
became 1,637 and the next run printed `3 of 3 day(s) already ingested`, exit 0.
This is the same shape as the midnight-straddle bug. **Fix:** each day is written
inside one transaction, so "started" and "finished" become the same question.

**2. Compaction could leave your database with zero tables, and say DONE ALL.**
On a nearly-full disk DuckDB cannot checkpoint on close and *does not raise*, so
compaction reported success while its rows were still only in
`psd_c.duckdb.wal`. The row-count guard was fooled too, because opening the build
file replays that WAL. `swap_in` then renamed only the `.duckdb`. Reproduced 3
for 3 in a 5–7 MB free-space window. **Fix:** explicit `CHECKPOINT` so the disk
error surfaces where it can be reported, plus a refusal to swap while either
`.wal` exists.

**3. A stale build file rolled the live database back.** Interrupt compaction,
let the normal monthly ingest add a day, compact again: the old build file's
`done` table made the second run skip every sensor it already had, call itself
complete, and swap the new days away. Measured 21,904 captures down to 20,348.
**Fix:** build files now carry a fingerprint of the source they were started
from, and a mismatch deletes them.

**4. `build_db.py` deleted `spectrum.duckdb` before rebuilding it.** So from the
moment the summaries step starts there is no zoomed-out layer — and if the run
dies, a 12 KB stub is all that is left of a 0.5 GB database. **This is what is
happening on your machine right now** (Part 3). **Fix:** it builds beside and
swaps, keeping a `.bak`, the way `compact_db.py` always has.

**5. A blank cell in a CSV stored full-scale power.** DuckDB returns a NULL
column as a masked array; `np.asarray` drops the mask and exposes the 0.0
underneath. 0.0 dBm/Hz is far above the ceiling, so it clipped to byte 255 — the
top of the colour scale, an invented emitter where the measurement was missing,
against a noise floor near −152. It then propagated into the max pyramid and
survived every zoom-out, with no warning (a few cells is under the old 1%
threshold). A blank *timestamp* stored `t = NaN`, which made `psd_meta`'s range
`NaN`, which is not valid JSON — so the browser could not parse `/api/psd_meta`
and the PSD layer silently never loaded. **Fix:** both refused with a reason.
None of your current exports contain blanks; this was latent.

### Also fixed

- **Ctrl+C did not stop the ingest.** DuckDB raises `InterruptException`, an
  ordinary `Exception`, so the per-file handler swallowed it and moved to the
  next day. Now it rolls the day back, says so, and exits.
- **A non-UTC timestamp offset made a resume duplicate every capture.** The
  resume check stripped the offset and read the digits as UTC while the ingest
  stored the real instant — 3 rows became 6, exit 0, invisible on the max layer
  and quietly double-weighting every mean and median window.
- **Two files claiming one sensor-day silently lost one,** decided by directory
  iteration order. Now reported, and the winner is deterministic.
- **A corrupt `spectrum.duckdb` killed the whole server** before it bound the
  port, taking PSD, PFP and IQ down with it. Now it degrades to "no summary
  layer" and says why.
- **serve.py picked its on-disk shape by table presence, not row count.** An
  empty `psd_chunk` beside a full row table advertised the layer and 404'd every
  tile; a database with data in both served only the chunked part as if that were
  everything (6 of 818 captures behind the picture, `gap: false`, 23.7 dB out).
- **A leaked index-worker slot could stall the in-RAM index permanently.**
  `Thread.start()` raises `MemoryError` under pressure, which escaped the
  `RuntimeError` handler; on a 2-core box that left zero workers for the life of
  the process. Also, a transient index failure is now retried after a cooldown
  instead of being a permanent verdict, and a *partial* pyramid no longer
  suppresses the in-RAM index (measured 13 dB worse than having no pyramid).
- **A header-only export counted as a day done** on a compacted database, so the
  real file for that day could never be picked up.
- **Environment plumbing:** relative `ATLAS_DB_DIR` silently split the pipeline
  across two databases; `compact_db.py` ignored `PSD_DB` and then advised you to
  build what you had just built; `serve.py` had no dependency guard; an ENOSPC in
  one layer aborted the run before the others were reached; unopenable database
  paths surfaced as raw tracebacks.

### Reported, deliberately not changed

- **`psd_lvl` is a max pyramid for all three statistics.** So a zoomed-out
  "median" view shows the largest per-sweep median in each column, not the median
  over that span — measured 3 to 11 dB high at one-day-per-pixel, worst bin
  22 dB. This is a deliberate display convention (the per-pixel pooling in
  `serve.py` does the same thing whether or not the pyramid exists, and turning
  the pyramid off only accounts for ~2 dB of it), and it is exact when you zoom
  in. But it was undocumented, so it is now written down in
  `build_psd_levels.py` and the manual with the measured magnitudes.
- **`PSD_DBM_OFFSET = 70.0` is correct** — `10·log10(10 MHz)`, converting
  dBm/Hz to channel power so the colours line up across the zoom boundary. Worth
  knowing it is not the power in the 80 kHz bin being drawn (that would be
  +49.03 dB).
- **The colour-match remap** shifts a displayed value up to ~3 dB so the PSD and
  summary layers agree at the handoff. Deliberate, disable-able with
  `ATLAS_NO_COLORMATCH=1`.
- **`_psd_count` over-estimates on the chunk schema**, and at `w ≤ 12` (grid
  columns ≤ 25) that can return the wrong capture's spectrum, up to 81 dB out.
  The viewer sends the plot width in pixels, so it never gets near this — it is
  an API-robustness gap. Left alone rather than touch the tile hot path.
- **The viewer draws frequency rows half a bin high** (+40 kHz), because the tile
  metadata carries bin *centres* and the rows are drawn between them. Sub-pixel
  except at maximum frequency zoom.
- **The legend's `vmax` reads 0.9–1.7 dB high** because the sampler picks chunks
  by timestamp quantile and re-picks the same chunk when there are few. Legend
  only.

---

## Test status

`python examples\test_all.py` — **11 of 11 suites pass**, 325 s, including a real
Chromium run of the viewer. Two suites are new:

- `test_durability.py`, every one a measured failure from this
  session: partial days under SIGKILL and SIGINT, orphaned WALs, stale build
  files, a destroyed summary rebuild, blank cells, blank timestamps, non-UTC
  offsets, sensor-day collisions, double compaction, mixed schemas, a corrupt
  summary database.
- `test_verify_1to1.py` — a real database damaged every way that matters, each
  damage required to be caught by a check that owns it.
