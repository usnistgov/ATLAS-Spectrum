"""test_fetch.py: prove ingest/fetch.py works, with no network access.

Spins up a throwaway HTTP server on localhost that speaks the same shape as the
NIST PDR record API (an RMM ResultData envelope whose components carry
downloadURL / filepath / size, plus .sha256 sidecars), points fetch.py at it
with ATLAS_PDR_BASE, and drives the real CLI as a subprocess.

Covers: source parsing for every documented form, the host policy and its
flags, listing modes, filtering, downloading, checksum verification, resume
after a truncated file, and the error messages for a host that is not there.

It also reproduces the failure that made real NIST downloads impossible. The
fixture runs a WAF in front of some of its paths that answers 403 to any client
that does not look like a browser, which is what data.nist.gov does. Those
cases pin down that a record behind bot protection now downloads, that the
header ladder walks far enough to satisfy a host wanting a different client,
that one blocked endpoint falls through to the next, and that when nothing gets
through the message says what to do instead.

    python examples/test_fetch.py          # exits 0 = PASS
"""

import hashlib
import http.server
import json
import math
import os
import shutil
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FETCH = os.path.join(ROOT, "ingest", "fetch.py")
sys.path.insert(0, os.path.join(ROOT, "ingest"))

failed = 0


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


# ---- the fake repository ----------------------------------------------

def signal(nsamp=32768, rate=2e6):
    """A real cf32_le capture with visible structure, 8 bytes per sample.

    Random bytes would exercise the download just as well, but reinterpreted as
    float32 they are full of NaNs and the STFT of that has no usable colour
    range. A single clean carrier is wrong in the other direction: it renders a
    nearly uniform tile that compresses to a few hundred bytes, which is
    indistinguishable from a broken render. So the fixture carries what a real
    capture carries -- a thermal noise floor, a steady carrier, a tone that
    steps frequency, and periodic bursts -- which is what lets one download be
    checked all the way through to a tile with real detail in it.

    Deliberately dependency-free and deterministic: this module is the network
    fixture, and it should not need numpy or produce different bytes per run.
    """
    seed = 12345

    def noise():                    # a plain LCG; nothing here needs quality
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed / 0x3FFFFFFF - 1.0

    out = bytearray()
    for i in range(nsamp):
        t = i / rate
        re = 0.03 * noise()
        im = 0.03 * noise()
        re += 0.35 * math.cos(2 * math.pi * 3.0e5 * t)
        im += 0.35 * math.sin(2 * math.pi * 3.0e5 * t)
        f2 = -6.0e5 + 2.0e5 * (i * 4 // nsamp)         # steps four times
        re += 0.25 * math.cos(2 * math.pi * f2 * t)
        im += 0.25 * math.sin(2 * math.pi * f2 * t)
        if (i // 512) % 3 == 0:                        # periodic bursts
            re += 0.30 * math.cos(2 * math.pi * 7.5e5 * t)
            im += 0.30 * math.sin(2 * math.pi * 7.5e5 * t)
        out += struct.pack("<ff", re, im)
    return bytes(out)


CAPTURE = signal()


# filepath -> bytes. Shaped like a real record: nested folders, a couple of
# formats, and enough files that the folder summary is the readable view.
BLOBS = {}
for _cfg in ("config_0", "config_1"):
    for _bw in ("1.4MHz", "5MHz"):
        BLOBS[f"{_bw}/{_cfg}/capture.sigmf-meta"] = json.dumps({
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 2e6,
                       "core:version": "1.0.0"},
            "captures": [{"core:sample_start": 0, "core:frequency": 2.412e9}],
            "annotations": [],
        }).encode()
        BLOBS[f"{_bw}/{_cfg}/capture.sigmf-data"] = CAPTURE
BLOBS["docs/readme.txt"] = b"a record-level readme\n"


def serve_fixture():
    """Start the fake repository on a free port. -> (server, port).

    Exposed so test_atlas.py can drive `atlas.py get <record>` against the same
    fixture rather than standing up a second one.
    """
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]

# Every Accept-Encoding the fixture was asked for. Byte accounting in fetch.py
# assumes an identity-coded body, so a request that allowed gzip would be a bug
# even if the download happened to succeed.
SEEN_ENCODINGS = set()

# Which file prefix each record's components point at. The prefixes differ in
# how they gate a request, so one fixture covers every rejection shape.
RECORD_PREFIX = {
    "mds2-test": "/files/",        # plain, no gating
    "mds2-alt": "/files/",         # same bytes, different hostname
    "mds2-waf": "/waf/",           # 403 unless the client looks like a browser
    "mds2-ladder": "/ladder/",     # 403 until the LAST header profile is tried
    "mds2-norange": "/norange/",   # 403 on a Range request, fine on a whole GET
}
BROWSERISH = "Mozilla/"
# fetch.py's honest self-identification, i.e. its final header profile.
PROJECT_CLIENT = "atlas-fetch"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _forbid(self):
        """What an edge WAF sends: a 403 with an HTML body, not a reason."""
        return self._send(403, b"<html><body>Access Denied</body></html>",
                          "text/html")

    def _manifest(self, rid, envelope=True):
        host = self.headers["Host"]
        if rid == "mds2-alt":
            # A different hostname for the files, so the test can allow the
            # manifest host and still refuse the file host.
            host = "localhost:" + host.split(":")[-1]
        base = f"http://{host}{RECORD_PREFIX[rid]}"
        comps = []
        for fp, data in sorted(BLOBS.items()):
            comps.append({"filepath": fp, "downloadURL": base + fp,
                          "size": len(data), "@type": ["nrdp:DataFile"]})
            comps.append({"filepath": fp + ".sha256",
                          "downloadURL": base + fp + ".sha256",
                          "size": 64, "@type": ["nrdp:ChecksumFile"]})
        comps.append({"@type": ["nrdp:Subcollection"], "filepath": "1.4MHz"})
        rec = {"title": "Synthetic test record", "components": comps}
        # The RMM search API wraps hits in an envelope; the /od/id resolver
        # returns the record bare. fetch.py has to accept both.
        doc = {"ResultCount": 1, "ResultData": [rec]} if envelope else rec
        return self._send(200, json.dumps(doc).encode(), "application/json")

    def _serve_blob(self, fp, allow_range=True):
        if fp.endswith(".sha256"):
            real = fp[:-len(".sha256")]
            if real not in BLOBS:
                return self._send(404, b"no")
            digest = hashlib.sha256(BLOBS[real]).hexdigest()
            return self._send(200, f"{digest}  {os.path.basename(real)}\n".encode())
        if fp not in BLOBS:
            return self._send(404, b"no")
        data = BLOBS[fp]
        rng = self.headers.get("Range")               # resume support
        if rng and rng.startswith("bytes="):
            if not allow_range:
                return self._forbid()
            start = int(rng[len("bytes="):].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data[start:])
            return
        return self._send(200, data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        ua = self.headers.get("User-Agent", "")
        SEEN_ENCODINGS.add(self.headers.get("Accept-Encoding") or "(none)")

        # An endpoint that is always blocked, to prove the record lookup falls
        # through to the next one instead of giving up on the first 403.
        if path.startswith("/blocked/records/"):
            return self._forbid()
        if path.startswith("/od/id/"):
            rid = path[len("/od/id/"):]
            if rid not in RECORD_PREFIX:
                return self._send(404, b"no", "text/plain")
            return self._manifest(rid, envelope=False)
        if path.startswith("/rmm/records/"):
            rid = path[len("/rmm/records/"):]
            if rid not in RECORD_PREFIX:
                return self._send(404, b'{"ResultCount":0,"ResultData":[]}',
                                  "application/json")
            # mds2-waf is gated the way data.nist.gov is: the manifest itself
            # is refused to a non-browser client, so the fetch used to die here
            # with "HTTP 403" before a single byte was downloaded.
            if rid == "mds2-waf" and BROWSERISH not in ua:
                return self._forbid()
            return self._manifest(rid)
        if path.startswith("/waf/"):
            if BROWSERISH not in ua:
                return self._forbid()
            return self._serve_blob(path[len("/waf/"):])
        if path.startswith("/ladder/"):
            # Only the honest project client is accepted, so getting a file out
            # of here requires walking the profiles to the very end.
            if PROJECT_CLIENT not in ua:
                return self._forbid()
            return self._serve_blob(path[len("/ladder/"):])
        if path.startswith("/norange/"):
            return self._serve_blob(path[len("/norange/"):], allow_range=False)
        if path.startswith("/files/"):
            return self._serve_blob(path[len("/files/"):])
        self._send(404, b"no")


def run(args, env_extra=None, expect=None):
    env = {**os.environ, **(env_extra or {})}
    p = subprocess.run([sys.executable, FETCH] + args, env=env,
                       capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    if expect is not None and p.returncode != expect:
        print(f"      (exit {p.returncode}, expected {expect})\n{out}")
    return p.returncode, out


def main():
    print("fetch.py offline test\n")

    # ---- 1. source parsing, in-process (no server needed) ----
    import fetch                                            # noqa: E402
    R = ("record", "mds2-3177")
    cases = {
        # ids
        "mds2-3177": R,
        "ark:/88434/mds2-3177": R,
        "ark:88434/mds2-3177": R,
        "doi:10.18434/mds2-3177": R,
        "10.18434/mds2-3177": R,
        # what the PDR landing page puts in the address bar
        "https://data.nist.gov/od/id/mds2-3177": R,
        "https://data.nist.gov/od/id/mds2-3177/": R,
        "https://data.nist.gov/od/id/ark:/88434/mds2-3177": R,
        "https://data.nist.gov/od/id/ark:/88434/mds2-3177/": R,
        "https://data.nist.gov/od/id/mds2-3177#files": R,
        "https://data.nist.gov/od/id/mds2-3177?selectedItemId=x": R,
        "https://data.nist.gov/rmm/records/mds2-3177": R,
        "https://data.nist.gov/rmm/records?@id=ark:/88434/mds2-3177": R,
        "https://doi.org/10.18434/mds2-3177": R,
        "http://doi.org/10.18434/mds2-3177": R,
        "http://data.nist.gov/od/id/mds2-3177": R,
        # the record-level download endpoint is a record, not a file
        "https://data.nist.gov/od/ds/mds2-3177": R,
        "https://data.nist.gov/od/ds/ark:/88434/mds2-3177": R,
        # scheme dropped by the paste
        "data.nist.gov/od/id/mds2-3177": R,
        "data.nist.gov/od/id/mds2-3177?x=1#f": R,
        # punctuation that rides along from a browser, chat or markdown
        "  https://data.nist.gov/od/id/mds2-3177  ": R,
        "<https://data.nist.gov/od/id/mds2-3177>": R,
        '"https://data.nist.gov/od/id/mds2-3177"': R,
        "'mds2-3177'": R,
        "https://data.nist.gov/od/id/mds2-3177,": R,
        "[mds2-3177](https://data.nist.gov/od/id/mds2-3177)": R,
        # direct file URLs, NIST and not
        "https://data.nist.gov/od/ds/ark:/88434/mds2-3177/1.4MHz/c0/x.tdms":
            ("file", "https://data.nist.gov/od/ds/ark:/88434/mds2-3177/1.4MHz/c0/x.tdms"),
        "https://nist-oar-cache.s3.amazonaws.com/x.tdms":
            ("file", "https://nist-oar-cache.s3.amazonaws.com/x.tdms"),
        "https://zenodo.org/records/1/files/x.npy":
            ("file", "https://zenodo.org/records/1/files/x.npy"),
    }
    bad = []
    for src, want in cases.items():
        try:
            got = fetch.parse_source(src)
        except Exception as e:                              # noqa: BLE001
            got = f"raised {type(e).__name__}"
        if got != want:
            bad.append(f"{src} -> {got}, wanted {want}")
    check(f"parse_source handles all {len(cases)} real-world paste forms",
          not bad, "; ".join(bad))

    # host policy
    default, strict = fetch.Policy(), fetch.Policy(nist_only=True)
    allowed = fetch.Policy(nist_only=True, extra_hosts=["zenodo.org"])
    check("default policy allows a non-NIST https host",
          default.allows("https://zenodo.org/x.npy"))
    check("default policy still refuses plaintext http",
          not default.allows("http://zenodo.org/x.npy"))
    check("--allow-http permits http",
          fetch.Policy(allow_http=True).allows("http://127.0.0.1:1/x"))
    check("--nist-only refuses a non-NIST host",
          not strict.allows("https://zenodo.org/x.npy"))
    check("--nist-only still allows the OAR cache bucket",
          strict.allows("https://nist-oar-cache.s3.amazonaws.com/x.tdms"))
    check("--allow-host re-permits a host under --nist-only",
          allowed.allows("https://zenodo.org/x.npy"))
    why = default.refusal("http://data.nist.gov/x")
    check("http:// refusal names the scheme, not the host",
          why is not None and "unencrypted" in why and "not a NIST host" not in why,
          (why or "").splitlines()[-1].strip())

    # ---- 2. drive the real CLI against a local fake PDR ----
    srv, port = serve_fixture()
    env = {"ATLAS_PDR_BASE": f"http://127.0.0.1:{port}/rmm/records/{{}}"}
    tmp = tempfile.mkdtemp(prefix="atlas-fetch-test-")
    try:
        rc, out = run(["mds2-test", "--list", "--allow-http"], env, expect=0)
        check("--list summarises by folder", rc == 0 and "folder(s):" in out
              and "docs/" in out and "9 file(s)" in out,
              out.strip().splitlines()[-1] if out else "")

        rc, out = run(["mds2-test", "--tree", "--allow-http"], env, expect=0)
        check("--tree prints the folder structure",
              rc == 0 and "1.4MHz/" in out and "config_0/" in out)

        rc, out = run(["mds2-test", "--long", "--allow-http",
                       "--filter", "1.4MHz/config_0"], env, expect=0)
        check("--long + --filter narrows to one folder",
              rc == 0 and out.count("/files/") == 2
              and "5MHz" not in out.replace("1.4MHz", ""), f"{out.count('/files/')} urls")

        rc, out = run(["mds2-test", "--filter", "nothing-matches",
                       "--allow-http"], env, expect=1)
        check("an empty filter result explains itself",
              rc == 1 and "run with --list" in out)

        dest = os.path.join(tmp, "iqdata")
        rc, out = run(["mds2-test", "--filter", "1.4MHz/config_0",
                       "--dest", dest, "--allow-http"], env, expect=0)
        got = sorted(os.listdir(os.path.join(dest, "1.4MHz", "config_0"))) \
            if os.path.isdir(os.path.join(dest, "1.4MHz", "config_0")) else []
        check("downloads preserve the record's folder structure",
              rc == 0 and got == ["capture.sigmf-data", "capture.sigmf-meta"],
              str(got))
        check("checksums are verified and the run reports 2 downloaded",
              "2 downloaded" in out and "failed" in out and "1 failed" not in out)
        check("it names the next command to run",
              "next: python ingest/iq_ingest.py" in out,
              next((l for l in out.splitlines() if l.startswith("next:")), ""))

        rc, out = run(["mds2-test", "--filter", "1.4MHz/config_0",
                       "--dest", dest, "--allow-http"], env, expect=0)
        check("a second run skips completed files", "2 already complete" in out)

        part = os.path.join(dest, "1.4MHz", "config_0", "capture.sigmf-data")
        with open(part, "r+b") as f:
            f.truncate(1000)
        rc, out = run(["mds2-test", "--filter", "1.4MHz/config_0",
                       "--dest", dest, "--allow-http"], env, expect=0)
        want = BLOBS["1.4MHz/config_0/capture.sigmf-data"]
        check("a truncated file resumes over HTTP Range",
              rc == 0 and os.path.getsize(part) == len(want)
              and want == open(part, "rb").read(),
              f"{os.path.getsize(part)} of {len(want)} bytes")

        flat = os.path.join(tmp, "flat")
        rc, out = run(["mds2-test", "--filter", "readme", "--dest", flat,
                       "--flat", "--allow-http"], env, expect=0)
        check("--flat writes basenames straight into --dest",
              rc == 0 and os.path.exists(os.path.join(flat, "readme.txt")))

        check("requests ask for an identity-coded body, so byte counts are real",
              SEEN_ENCODINGS == {"identity"}, str(sorted(SEEN_ENCODINGS)))

        # --first is the "just try it" path, and a .sigmf-meta without its
        # .sigmf-data is a download that succeeds and then ingests nothing.
        first = os.path.join(tmp, "first")
        rc, out = run(["mds2-test", "--first", "2", "--dest", first,
                       "--allow-http"], env, expect=0)
        pulled = sorted(os.path.relpath(os.path.join(dp, n), first)
                        .replace("\\", "/")
                        for dp, _d, ns_ in os.walk(first) for n in ns_)
        metas = [p for p in pulled if p.endswith(".sigmf-meta")]
        orphans = [m for m in metas
                   if m[:-len(".sigmf-meta")] + ".sigmf-data" not in pulled]
        check("--first keeps every capture whole instead of taking bare metadata",
              rc == 0 and metas and not orphans, str(pulled))
        check("--first still counts units, not files, in what it reports",
              "smallest of" in out and "capture(s)" in out,
              next((l.strip() for l in out.splitlines() if "smallest of" in l), ""))

        # ---- 3. the 403 that stopped real NIST downloads ----
        # mds2-waf refuses both its manifest and its files to any client that
        # does not look like a browser. This is the exact shape of the failure
        # reported against data.nist.gov: HTTP 403, download never starts,
        # nothing ingested.
        wafdir = os.path.join(tmp, "waf")
        rc, out = run(["mds2-waf", "--filter", "1.4MHz/config_0",
                       "--dest", wafdir, "--allow-http"], env, expect=0)
        got = sorted(os.listdir(os.path.join(wafdir, "1.4MHz", "config_0"))) \
            if os.path.isdir(os.path.join(wafdir, "1.4MHz", "config_0")) else []
        check("a record behind bot protection downloads instead of 403ing",
              rc == 0 and got == ["capture.sigmf-data", "capture.sigmf-meta"]
              and "2 downloaded" in out, str(got) or out.strip()[-160:])

        # /ladder/ accepts only the last profile, so the file arrives only if
        # every profile is tried rather than the first rejection being fatal.
        lad = os.path.join(tmp, "ladder")
        rc, out = run(["mds2-ladder", "--filter", "readme", "--dest", lad,
                       "--allow-http"], env, expect=0)
        check("the header ladder keeps going until a profile is accepted",
              rc == 0 and os.path.exists(os.path.join(lad, "docs", "readme.txt"))
              and "retrying as a different client" in out,
              out.strip().splitlines()[-1] if out else "")

        # A resume that is refused must cost the resume, not the file.
        nr = os.path.join(tmp, "norange")
        rc, out = run(["mds2-norange", "--filter", "1.4MHz/config_0",
                       "--dest", nr, "--allow-http"], env, expect=0)
        part = os.path.join(nr, "1.4MHz", "config_0", "capture.sigmf-data")
        check("the no-range fixture downloads whole on a first pass", rc == 0
              and os.path.exists(part))
        with open(part, "r+b") as f:
            f.truncate(1000)
        rc, out = run(["mds2-norange", "--filter", "1.4MHz/config_0",
                       "--dest", nr, "--allow-http"], env, expect=0)
        want = BLOBS["1.4MHz/config_0/capture.sigmf-data"]
        check("a 403 on the resume request restarts instead of failing",
              rc == 0 and open(part, "rb").read() == want
              and "restarting from the beginning" in out,
              f"{os.path.getsize(part)} of {len(want)} bytes")

        # Nothing gets through: the run must still say what to do next.
        rc, out = run(["mds2-waf", "--list", "--allow-http"],
                      {**env, "ATLAS_USER_AGENT": "some-scraper/1.0"}, expect=1)
        check("a hard 403 explains the cause and the way round it",
              rc == 1 and "Traceback" not in out and "--user-agent" in out
              and "python atlas.py get" in out and "bot protection" in out,
              out.strip().splitlines()[-1] if out else "")

        rc, out = run(["mds2-waf", "--list", "--allow-http",
                       "--user-agent", fetch.BROWSER_UA], env, expect=0)
        check("--user-agent is honoured and gets the same record listed",
              rc == 0 and "folder(s):" in out)

        # ---- 4. one blocked endpoint must not sink the lookup ----
        # /blocked/ always 403s; /od/id/ answers with a bare NERDm record. In
        # process, because this is about the endpoint list, not the CLI.
        base = f"http://127.0.0.1:{port}"
        fetch.POLICY = fetch.Policy(allow_http=True)
        saved = os.environ.pop("ATLAS_PDR_BASE", None)
        try:
            fetch.RECORD_ENDPOINTS = (base + "/blocked/records/{}",
                                      base + "/od/id/{}")
            rec = fetch.fetch_record("mds2-test", retries=0)
            check("a 403 on the first endpoint falls through to the next",
                  isinstance(rec.get("components"), list)
                  and len(rec["components"]) > 1,
                  f"{len(rec.get('components', []))} components")
            check("a bare NERDm record parses as well as an RMM envelope",
                  fetch.unwrap_record(rec) is rec)

            fetch.RECORD_ENDPOINTS = (base + "/blocked/records/{}",)
            try:
                fetch.fetch_record("mds2-test", retries=0)
                check("every endpoint blocked raises FetchError", False)
            except fetch.FetchError as e:
                check("every endpoint blocked names each attempt and the fix",
                      "/blocked/records/mds2-test" in str(e)
                      and "python atlas.py get" in str(e),
                      str(e).splitlines()[0])

            fetch.RECORD_ENDPOINTS = (base + "/rmm/records/{}",
                                      base + "/od/id/{}")
            try:
                fetch.fetch_record("mds2-nosuch", retries=0)
                check("a record missing everywhere raises FetchError", False)
            except fetch.FetchError as e:
                check("404 everywhere still reads as 'no such record'",
                      "no record" in str(e) and "not be published yet" in str(e),
                      str(e).splitlines()[0])
        finally:
            if saved is not None:
                os.environ["ATLAS_PDR_BASE"] = saved

        rc, out = run(["mds2-test", "--list"], env, expect=1)
        check("without --allow-http the http fixture is refused clearly",
              rc == 1 and "unencrypted" in out and "--allow-http" in out)

        rc, out = run(["mds2-test", "--list", "--allow-http", "--nist-only"],
                      env, expect=1)
        check("--nist-only refuses the record host and names the flag",
              rc == 1 and "not a NIST host" in out
              and "--allow-host 127.0.0.1" in out)

        rc, out = run(["mds2-alt", "--list", "--allow-http", "--nist-only",
                       "--allow-host", "127.0.0.1"], env, expect=1)
        check("--nist-only drops off-host components and says how to override",
              rc == 1 and "dropped by the host policy" in out
              and "--allow-host localhost" in out,
              next((l for l in out.splitlines() if "dropped" in l), ""))

        alt = os.path.join(tmp, "alt")
        rc, out = run(["mds2-alt", "--filter", "readme", "--dest", alt,
                       "--allow-http"], env, expect=0)
        check("by default those same components download fine",
              rc == 0 and os.path.exists(
                  os.path.join(alt, "docs", "readme.txt"))
              and "note: localhost is not a NIST host" in out)

        rc, out = run(["mds2-nosuch", "--list", "--allow-http"], env, expect=1)
        check("a missing record gives one sentence, not a traceback",
              rc == 1 and "Traceback" not in out and "no record" in out)

        rc, out = run(["--retries", "0",
                       "https://not-a-real-host.invalid/x.tdms"], None, expect=1)
        check("an unreachable host gives one sentence, not a traceback",
              rc == 1 and "Traceback" not in out
              and ("cannot resolve" in out or "cannot reach" in out),
              out.strip().splitlines()[-1] if out else "")

        # ---- 3. TLS failures: permanent, and told apart ----
        # A rejected certificate used to be classed transient (ssl.SSLError
        # inherits OSError), so every one burned the full ladder -- three
        # retries across three header profiles -- and then blamed a proxy
        # whatever the actual reason was. Both halves are checked here: that
        # it is not retried, and that the three verification failures give
        # three different answers, since the fix for each is different.
        def tls(msg, verify=None):
            e = ssl.SSLCertVerificationError(1, msg)
            e.reason = "CERTIFICATE_VERIFY_FAILED"
            if verify:
                e.verify_message = verify
            return urllib.error.URLError(e)

        expired = tls("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
                      "failed: certificate has expired (_ssl.c:1081)",
                      "certificate has expired")
        selfsigned = tls("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
                         "failed: self signed certificate in certificate chain",
                         "self signed certificate in certificate chain")
        mismatch = tls("hostname 'a' doesn't match 'b'", "Hostname mismatch")

        check("a rejected certificate is never retried",
              not any(fetch._is_transient(e)
                      for e in (expired, selfsigned, mismatch)))
        check("a dropped connection still is retried",
              fetch._is_transient(urllib.error.URLError(
                  ConnectionResetError("reset"))))

        # "certifi" is a substring of "certificate", so the marker for the
        # expired branch has to be the whole pip command.
        UPGRADE = "upgrade certifi"
        url = "https://data.nist.gov/od/ds/mds2-3177/x.csv"
        msg = str(fetch.net_error(url, expired))
        check("an expired certificate points at the local CA bundle",
              UPGRADE in msg and "data.nist.gov" in msg
              and "SSL_CERT_FILE" not in msg,
              msg.splitlines()[1].strip()[:64])
        msg = str(fetch.net_error(url, selfsigned))
        check("a self-signed chain points at the proxy instead",
              "SSL_CERT_FILE" in msg and UPGRADE not in msg,
              msg.splitlines()[1].strip()[:64])
        msg = str(fetch.net_error(url, mismatch))
        check("a hostname mismatch says the certificate is for another host",
              "not valid for data.nist.gov" in msg and UPGRADE not in msg,
              msg.splitlines()[1].strip()[:64])
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - fetch.py verified offline against a local fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
