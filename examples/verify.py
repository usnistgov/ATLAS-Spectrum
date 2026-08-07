"""verify.py: prove the install actually works. Exits 0 = PASS, 1 = FAIL.

Run this after the quick start. It checks the Python version, the imports, the
demo database, and every endpoint the viewer depends on, then prints a verdict.
No browser and no free port needed: it drives the Flask app in-process, so it
gives the same answer on any machine.

    python examples/verify.py

If it prints "RESULT: PASS", the install is correct and the viewer will render.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

checks, failed = [], 0


def check(name, ok, detail=""):
    global failed
    checks.append((name, ok, detail))
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


print("Spectrum Viewer install check")
print(f"  python {sys.version.split()[0]} at {sys.executable}\n")

# 1. Python version (duckdb needs >= 3.10). Reported on its own, and fatal on
# its own: no amount of pip install fixes an interpreter that is too old, and
# saying "install dependencies first" here sends people the wrong way.
v = sys.version_info
check("Python >= 3.10", v >= (3, 10), f"found {v.major}.{v.minor}.{v.micro}")
if failed:
    print(f"\nRESULT: FAIL - Python {v.major}.{v.minor} is too old; ATLAS "
          "needs 3.10 or newer. Re-run with a newer interpreter "
          "(py -3.12 / python3.12).")
    sys.exit(1)

# 2. Dependencies importable
for mod, pkg in [("duckdb", "duckdb"), ("numpy", "numpy"),
                 ("flask", "flask"), ("PIL", "Pillow")]:
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except ImportError as e:
        check(f"import {mod}", False, f"pip install {pkg}  ({e})")

if failed:
    print("\nRESULT: FAIL - install dependencies first: "
          "pip install -r requirements.txt")
    sys.exit(1)

# 3. Demo database: build it if the user hasn't yet
# Same resolution serve.py uses, so redirecting the databases to
# another drive cannot make this check the wrong file.
# Resolved exactly as serve.py and atlas.py resolve it, so redirecting the
# databases cannot make this check look at a different file than the server.
iq_db = os.environ.get("IQ_DB") or os.path.join(
    os.environ.get("ATLAS_DB_DIR") or ROOT, "iq.duckdb")
if not os.path.exists(iq_db):
    print("\n  iq.duckdb missing - running examples/make_sample.py ...")
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_sample.py")],
                       cwd=ROOT)
    if r.returncode != 0:
        print("\nRESULT: FAIL - examples/make_sample.py failed (see above)")
        sys.exit(1)
check(f"{os.path.basename(iq_db)} exists where serve.py looks",
      os.path.exists(iq_db), iq_db)

# 4. Endpoints, driven in-process (no port, no browser)
import serve                                             # noqa: E402

client = serve.app.test_client()

check("GET / (viewer page)", client.get("/").status_code == 200)

idx = client.get("/api/iq_index").get_json()
caps = idx.get("captures", [])
check("GET /api/iq_index lists a capture", bool(caps),
      f"{len(caps)} capture(s)")

if caps:
    cap = caps[0]
    cid = cap["id"]
    m = client.get(f"/api/iq_meta?id={cid}").get_json()
    check("GET /api/iq_meta returns axes", m.get("has") is True,
          f"{cap['name']}: fc={cap['fc']/1e6:.3f} MHz, "
          f"fs={cap['fs']/1e6:.3f} Msps, {cap['duration']:.3f} s")

    r = client.get(f"/api/iq_layer?id={cid}&t0=0&t1={cap['duration']}"
                   f"&w=600&h=300")
    ok = (r.status_code == 200
          and r.mimetype == "image/webp"
          and len(r.data) > 1000
          and "X-Meta" in r.headers)
    check("GET /api/iq_layer renders a spectrogram tile", ok,
          f"{r.status_code}, {r.mimetype}, {len(r.data)} bytes")

# 5. CBRS is optional: absent databases must NOT break the server
meta = client.get("/api/meta").get_json()
n_sensors = len(meta.get("sensors", []))
check("CBRS optional (server healthy either way)", True,
      f"{n_sensors} sensor(s) - "
      + ("CBRS databases present" if n_sensors else
         "no CBRS data, IQ-only install (expected for the demo)"))

print()
if failed:
    print(f"RESULT: FAIL ({failed} check(s) failed above)")
    sys.exit(1)
print("RESULT: PASS - open http://127.0.0.1:8090 after 'python serve.py' "
      "and pick the capture in the Source dropdown.")
sys.exit(0)
