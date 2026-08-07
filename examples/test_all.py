"""test_all.py: run every test in the project and print one verdict.

Nothing here touches the network or the databases beside serve.py: each suite
builds its own fixtures in a temporary folder, and the download tests run
against a fake NIST repository served on localhost. So this gives the same
answer on any machine, online or off.

    python examples/test_all.py            # exits 0 = everything passes
    python examples/test_all.py -q         # only the per-suite verdicts

What each one covers:

    verify.py       the install: version, imports, demo database, endpoints
    test_fetch.py   downloading: source parsing, host policy, resume,
                    checksums, and the HTTP 403 that bot protection causes
    test_ingest.py  the CBRS path end to end, compacted and uncompacted
    test_durability.py  the ways an interrupted or damaged run used to lose
                    data silently: partial days, orphaned WALs, stale build
                    files, blank cells, non-UTC stamps
    test_verify_1to1.py  damages a real database every way the ingest could
                    plausibly get one wrong -- including a round-half-up
                    quantizer, which a half-step tolerance cannot see -- and
                    requires each to be caught by a check that owns it
    test_repeat.py  re-running the ingest months later: additive, idempotent,
                    and safe on an already-compacted database
    test_ingest_all.py  the one-command ingest over an unfamiliar folder, and
                    every way a re-run used to destroy data
    test_atlas.py   `atlas.py get` on every kind of data, including a record
                    downloaded and ingested through to a rendered tile
    test_cli.py     every command, flag, prompt input and menu choice
    test_serve.py   every endpoint, and every control the viewer drives
    test_viewer.py  viewer.html in a real browser, pressing every control.
                    Optional: skips cleanly without playwright + a Chromium
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Cheapest and most fundamental first: if the install is broken there is no
# point running the rest, and if fetching is broken the download tests explain
# themselves better than the suites built on top of them.
SUITES = [
    ("verify.py", "install and endpoints"),
    ("test_fetch.py", "downloading, including the 403 path"),
    ("test_ingest.py", "the CBRS ingest path"),
    ("test_repeat.py", "re-running the ingest as new data arrives"),
    ("test_durability.py", "interrupted runs, full disks and damaged files"),
    ("test_verify_1to1.py", "the 1:1 verifier can actually fail"),
    ("test_ingest_all.py", "one command over any folder, and re-running it"),
    ("test_atlas.py", "atlas.py get, on data and on records"),
    ("test_cli.py", "every command, flag and menu choice"),
    ("test_serve.py", "every endpoint and viewer control"),
    # Last because it is the only one that needs anything the project does not
    # already require. It reports SKIP, and exits 0, when it cannot run.
    ("test_viewer.py", "the viewer in a real browser (optional)"),
]


def main():
    quiet = "-q" in sys.argv or "--quiet" in sys.argv
    print(f"ATLAS test suite\n  python {sys.version.split()[0]} at "
          f"{sys.executable}\n  repository {ROOT}\n")
    # Keep the suites out of the working copy: they each redirect the databases
    # anyway, but a stray run should never overwrite someone's real data.
    env = dict(os.environ)
    results, total = [], 0.0
    for script, what in SUITES:
        t0 = time.time()
        print(f"---- {script}: {what}", flush=True)
        p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           capture_output=quiet, text=True)
        dt = time.time() - t0
        total += dt
        ok = p.returncode == 0
        results.append((script, ok, dt))
        if quiet:
            out = (p.stdout or "") + (p.stderr or "")
            verdict = next((l for l in reversed(out.splitlines())
                            if l.startswith("RESULT:")), "")
            word = "PASS" if ok else "FAIL"
            if ok and verdict.startswith("RESULT: SKIP"):
                word = "SKIP"
            print(f"     {word}  {dt:5.1f}s  {verdict}")
            if not ok:
                for line in out.splitlines():
                    if "[FAIL]" in line:
                        print(f"       {line.strip()}")
        else:
            print(f"---- {script}: {'PASS' if ok else 'FAIL'} in {dt:.1f}s\n",
                  flush=True)

    bad = [s for s, ok, _ in results if not ok]
    print("=" * 62)
    for script, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {dt:6.1f}s  {script}")
    print(f"\n{len(results) - len(bad)} of {len(results)} suite(s) passed "
          f"in {total:.0f}s")
    if bad:
        print(f"\nRESULT: FAIL - {', '.join(bad)}. Re-run one on its own for "
              "the detail:\n    python examples/" + bad[0])
        return 1
    print("\nRESULT: PASS - every suite passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
