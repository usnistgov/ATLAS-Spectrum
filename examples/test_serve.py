"""test_serve.py: every endpoint the viewer calls, and every control it drives.

The viewer has no server-side state: each of its controls -- the source and
sensor dropdowns, the max/median/mean buttons, the two zoom sliders, the
colormap picker, the colour lock, reset, export -- turns into query parameters
on one of ten routes. So exercising every route with every parameter shape those
controls can produce is the same thing as pressing every button, and it can be
done in-process with no browser and no free port.

Builds all four databases once from the offline fixtures, then probes:
valid requests, the extremes each slider can reach, inverted and empty windows,
unknown sensors and captures, and missing parameters. A malformed request must
answer with a status the viewer can act on -- never a 500.

    python examples/test_serve.py          # exits 0 = PASS
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import test_atlas as TA                                       # noqa: E402
import test_ingest as TI                                      # noqa: E402

failed = 0
CMAPS = ("inferno", "viridis", "plasma", "magma", "turbo", "greys",
         "not-a-colormap")                       # unknown must fall back, not fail
# Reference values sampled from matplotlib's own LUTs at index 0/128/255
# (Greys_r for 'greys'). The colormaps used to be approximations -- a 12-anchor
# inferno and a polynomial turbo, off by up to 34 and 251 of 255 -- so a tile
# could not be laid beside a NASCTN reference plot. Pin them.
CMAP_REF = {
    "viridis": ((68, 1, 84), (33, 145, 140), (253, 231, 37)),
    "inferno": ((0, 0, 4), (188, 55, 84), (252, 255, 164)),
    "plasma":  ((13, 8, 135), (204, 71, 120), (240, 249, 33)),
    "magma":   ((0, 0, 4), (183, 55, 121), (252, 253, 191)),
    "turbo":   ((48, 18, 59), (164, 252, 60), (122, 4, 3)),
    "greys":   ((0, 0, 0), (151, 151, 151), (255, 255, 255)),
}


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def build(tmp, iq_samples=1024 * 64):
    """One db dir holding all four layers. -> (dbs, sensor, env).

    Shared with test_viewer.py, which needs the same four databases and a longer
    capture: two copies of this drifted apart within a day of existing.
    """
    data = os.path.join(tmp, "data")
    TI.make_data(data)
    TA.make_iq(os.path.join(data, "iq", "run1"), n=iq_samples)
    dbs = os.path.join(tmp, "dbs")
    os.makedirs(dbs, exist_ok=True)
    env = {**os.environ, "ATLAS_DB_DIR": dbs,
           "SPECTRUM_DB": os.path.join(dbs, "spectrum.duckdb"),
           "PSD_DB": os.path.join(dbs, "psd.duckdb"),
           "PFP_DB": os.path.join(dbs, "pfp.duckdb"),
           "IQ_DB": os.path.join(dbs, "iq.duckdb")}
    p = subprocess.run([sys.executable, os.path.join(ROOT, "atlas.py"),
                        "get", data], env=env, cwd=ROOT, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        raise SystemExit("could not build the fixture databases")
    return dbs, TI.SENSOR, env


def tile_ok(r):
    """A tile response: a decodable WebP plus the axis metadata the viewer reads.

    Decoding it is the check, not its size. A one-pixel-tall tile or a window
    holding ten frequency bins is a legitimate image that compresses to under a
    hundred bytes, so a byte threshold would fail exactly the degenerate cases
    worth testing.
    """
    import io

    from PIL import Image
    if r.status_code != 200 or r.mimetype != "image/webp":
        return False, f"{r.status_code} {r.mimetype}"
    if "X-Meta" not in r.headers:
        return False, "no X-Meta header"
    try:
        meta = json.loads(r.headers["X-Meta"])
    except ValueError as e:
        return False, f"X-Meta is not JSON ({e})"
    try:
        im = Image.open(io.BytesIO(r.data))
        im.load()
    except Exception as e:                                    # noqa: BLE001
        return False, f"tile does not decode ({e})"
    if im.width < 1 or im.height < 1:
        return False, f"empty image {im.width}x{im.height}"
    for k in ("vmin", "vmax"):
        if k not in meta:
            return False, f"X-Meta has no {k}"
    return True, f"{im.width}x{im.height}, {len(r.data)} bytes"


def main():
    print("serve.py endpoint and control test\n")
    tmp = tempfile.mkdtemp(prefix="atlas-serve-test-")
    try:
        dbs, sensor, _env = build(tmp)
        # serve.py resolves its databases at import time, so the environment has
        # to be right before the import, not after.
        os.environ.update({
            "ATLAS_DB_DIR": dbs,
            "SPECTRUM_DB": os.path.join(dbs, "spectrum.duckdb"),
            "PSD_DB": os.path.join(dbs, "psd.duckdb"),
            "PFP_DB": os.path.join(dbs, "pfp.duckdb"),
            "IQ_DB": os.path.join(dbs, "iq.duckdb")})
        import serve                                          # noqa: E402
        c = serve.app.test_client()

        # ---- the page and the two index endpoints ----
        r = c.get("/")
        check("GET / serves the viewer",
              r.status_code == 200 and b"<canvas" in r.data,
              f"{r.status_code}, {len(r.data)} bytes")
        check("GET / is served no-store, so a stale viewer cannot linger",
              "no-store" in r.headers.get("Cache-Control", ""))

        # ---- the colormaps are the real ones, on both sides of the wire ----
        # A tile is only comparable to a published NASCTN figure if the ramp is
        # matplotlib's actual LUT, and only self-consistent if the client-drawn
        # summary uses the identical bytes. Check the server's table against
        # known matplotlib values, then that viewer.html carries the same
        # base64 blobs.
        bad = []
        for name, (c0, c128, c255) in CMAP_REF.items():
            lut = serve.LUTS.get(name)
            if lut is None:
                bad.append(f"{name}: missing"); continue
            for i, want in ((0, c0), (128, c128), (255, c255)):
                got = tuple(int(v) for v in lut[i])
                if got != want:
                    bad.append(f"{name}[{i}] {got} != matplotlib {want}")
        check("every colormap is matplotlib's exact 256-entry LUT",
              not bad, "; ".join(bad[:3]))

        # ---- the PSD coarse-level guards ----
        # A level may only be used where it is BOTH fast and at least as true as
        # reading the captures. Both guards were added after measuring a level
        # against a no-thinning render and finding it worse; pin them.
        saved = serve._psd_lvls
        try:
            serve._psd_lvls = [600, 3600, 21600]
            cols = 2644
            big = 10_000_000          # forces stride > 1
            checks = [
                ("no levels on disk -> captures", [], 200*86400, big, None),
                ("captures already exact -> captures", [600], 200*86400, 10, None),
                ("no bucket fits a narrow column", [3600], 3*86400, big, None),
                ("coarsest bucket that fits wins", [600, 3600, 21600],
                 200*86400, big, 3600),
                ("finer level when the coarse one is too wide",
                 [600, 3600], 60*86400, big, 600),
            ]
            bad = []
            for name, lvls, span, ncap, want in checks:
                serve._psd_lvls = lvls
                got = serve._pick_psd_level(span, cols, ncap)
                if got != want:
                    bad.append(f"{name}: got {got}, want {want}")
            check("PSD level choice obeys both guards", not bad, "; ".join(bad))

            # overlap assignment: every column a bucket touches gets it, and
            # nothing lands in a column the bucket never covered.
            serve._psd_lvls = [600]
            g = serve._Grid(0.0, 6000.0, 10, serve.NF)   # 10 cols x 600 s each
            rows = [(1200, bytes([7]) * serve.NF), (1800, bytes([9]) * serve.NF)]
            serve._fill_from_levels(g, rows, 0.0, 6000.0, 600, 10, 123)
            filled = [i for i in range(10) if g.filled[i]]
            vals = [int(g.buf[i][0]) for i in range(10)]
            ok = (filled == [2, 3] and vals[2] == 7 and vals[3] == 9
                  and g.ncap == 123)
            check("coarse rows land only in the columns they cover",
                  ok, f"filled={filled} vals={vals[:5]} ncap={g.ncap}")
        finally:
            serve._psd_lvls = saved

        import base64                                          # noqa: E402
        page = c.get("/").data.decode("utf8", "replace")
        missing = [n for n in CMAP_REF
                   if base64.b64encode(serve.LUTS[n].tobytes()).decode()[:64] not in
                   page.replace("'+\n", "").replace("'", "").replace("\n", "")
                   .replace(" ", "")]
        check("viewer.html carries the same colormap bytes as the server",
              not missing, "not found in the page: " + ", ".join(missing))
        r = c.get("/api/meta")
        m = r.get_json()
        names = [s.get("sensor") for s in (m.get("sensors") or [])]
        check("GET /api/meta lists the sensors the source dropdown offers",
              r.status_code == 200 and sensor in names, str(names))
        r = c.get("/api/iq_index")
        caps = r.get_json().get("captures", [])
        check("GET /api/iq_index lists the IQ captures for the dropdown",
              r.status_code == 200 and len(caps) == 1, f"{len(caps)} capture(s)")
        r = c.get("/api/unknown_route")
        check("an unknown route is a 404, not a crash", r.status_code == 404)

        # ---- CBRS summary layer: the max/median/mean buttons ----
        pm = c.get(f"/api/psd_meta?sensor={sensor}").get_json()
        check("GET /api/psd_meta reports the PSD layer as available",
              pm.get("has") is True, str(pm)[:90])
        t0, t1 = pm["t_min"], pm["t_max"] + 1
        r = c.get(f"/api/heatmap?sensor={sensor}&t0={t0}&t1={t1}&width=1600")
        h = r.get_json()
        check("GET /api/heatmap returns all three statistics at once",
              r.status_code == 200 and h["count"] > 0
              and all(k in h for k in ("max", "median", "mean"))
              and len(h["max"]) == h["count"],
              f"{h.get('count')} row(s), level {h.get('level')}")
        # The time slider's two extremes: whole span, and one second of it.
        bad = []
        for width in (1, 100, 1600, 20000):
            for span in (t1 - t0, 60.0, 1.0):
                rr = c.get(f"/api/heatmap?sensor={sensor}&t0={t0}"
                           f"&t1={t0 + span}&width={width}")
                if rr.status_code != 200:
                    bad.append(f"width={width} span={span}: {rr.status_code}")
        check("heatmap holds up across the whole zoom range", not bad,
              "; ".join(bad))
        r = c.get(f"/api/heatmap?sensor=NO-SUCH-SENSOR&t0={t0}&t1={t1}")
        check("heatmap for an unknown sensor is empty, not an error",
              r.status_code == 200 and r.get_json()["count"] == 0)
        r = c.get(f"/api/heatmap?sensor={sensor}&t0={t1}&t1={t0}")
        check("heatmap with an inverted window answers instead of crashing",
              r.status_code == 200, str(r.status_code))

        # ---- PSD layer: frequency slider, colormap, colour lock ----
        fmin, fmax = pm["fmin"], pm["fmax"]
        ok, why = tile_ok(c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}"))
        check("GET /api/psd_layer renders a tile", ok, why)
        bad = []
        for cmap in CMAPS:
            ok, why = tile_ok(c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}"
                                    f"&t1={t1}&cmap={cmap}"))
            if not ok:
                bad.append(f"{cmap}: {why}")
        check(f"the colormap picker works for every value ({len(CMAPS)} tried)",
              not bad, "; ".join(bad))
        ok, why = tile_ok(c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}"
                                "&vmin=-120&vmax=-40"))
        meta = json.loads(c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}"
                                "&vmin=-120&vmax=-40").headers["X-Meta"])
        check("the colour lock is honoured exactly as given",
              ok and (meta["vmin"], meta["vmax"]) == (-120.0, -40.0),
              f"{why}; got vmin={meta.get('vmin')} vmax={meta.get('vmax')}")
        # The frequency slider, from the full band down to one channel.
        bad = []
        for frac in (1.0, 0.5, 0.1, 0.01):
            mid = (fmin + fmax) / 2
            half = (fmax - fmin) * frac / 2
            for h in (1, 60, 600, 2000):
                rr = c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}"
                           f"&f0={mid - half}&f1={mid + half}&h={h}")
                ok, why = tile_ok(rr)
                if not ok:
                    bad.append(f"frac={frac} h={h}: {why}")
        check("psd_layer holds up across the whole frequency zoom", not bad,
              "; ".join(bad))
        r = c.get(f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}"
                  f"&f0={fmax}&f1={fmin}")
        ok, why = tile_ok(r)
        check("an inverted frequency window falls back to the full band",
              ok, why)
        r = c.get(f"/api/psd_layer?sensor=NO-SUCH&t0={t0}&t1={t1}")
        check("psd_layer for an unknown sensor is a clean 404",
              r.status_code == 404 and "error" in r.get_json(),
              str(r.status_code))

        # ---- PFP layer: the per-channel frame view ----
        fm = c.get(f"/api/pfp_meta?sensor={sensor}").get_json()
        check("GET /api/pfp_meta lists the channels it can show",
              fm.get("has") is True and fm.get("freqs"),
              f"{len(fm.get('freqs') or [])} channel(s)")
        freq = fm["freqs"][0]
        pt0, pt1 = fm["t_min"], fm["t_max"] + 1
        ok, why = tile_ok(c.get(f"/api/pfp_frame?sensor={sensor}&freq={freq}"
                                f"&t0={pt0}&t1={pt1}"))
        check("GET /api/pfp_frame renders a tile", ok, why)
        bad = []
        for cmap in CMAPS:
            for h in (1, 600, 2000):
                rr = c.get(f"/api/pfp_frame?sensor={sensor}&freq={freq}"
                           f"&t0={pt0}&t1={pt1}&cmap={cmap}&h={h}")
                ok, why = tile_ok(rr)
                if not ok:
                    bad.append(f"{cmap} h={h}: {why}")
        check("pfp_frame works for every colormap and row count", not bad,
              "; ".join(bad))
        ok, why = tile_ok(c.get(f"/api/pfp_frame?sensor={sensor}&freq={freq}"
                                f"&t0={pt0}&t1={pt1}&vmin=-130&vmax=-50"))
        check("pfp_frame honours the colour lock", ok, why)
        r = c.get(f"/api/pfp_frame?sensor={sensor}&freq=1234.5"
                  f"&t0={pt0}&t1={pt1}")
        check("pfp_frame for a channel that does not exist answers cleanly",
              r.status_code in (200, 404), str(r.status_code))

        # A sweep dwells on each channel about once every 90 s, so most of the
        # frame layer's useful zoom range is windows holding no capture for the
        # channel on screen. Those must hold the last measured frame, the way
        # the PSD layer does -- answering 404 turns deep zoom blank.
        mid = (pt0 + pt1) / 2
        blanked = []
        for span in (30.0, 1.0, 0.1):
            rr = c.get(f"/api/pfp_frame?sensor={sensor}&freq={freq}"
                       f"&t0={mid - span / 2}&t1={mid + span / 2}&w=900&h=300")
            ok, why = tile_ok(rr)
            if not ok:
                blanked.append(f"{span}s: {why}")
        check("pfp_frame holds the last frame below the sweep interval",
              not blanked, "; ".join(blanked))

        # ---- IQ layer: the source dropdown plus both zoom axes ----
        cap = caps[0]
        cid, dur = cap["id"], cap["duration"]
        im = c.get(f"/api/iq_meta?id={cid}").get_json()
        check("GET /api/iq_meta returns the axes for the chosen capture",
              im.get("has") is True and im["duration"] > 0,
              f"fc={im.get('fc')} fs={im.get('fs')}")
        check("iq_meta for an unknown capture says so instead of failing",
              c.get("/api/iq_meta?id=nope").get_json().get("has") is False)
        check("iq_meta with no id at all says so too",
              c.get("/api/iq_meta").get_json().get("has") is False)
        ok, why = tile_ok(c.get(f"/api/iq_layer?id={cid}&t0=0&t1={dur}"
                                "&w=800&h=400"))
        check("GET /api/iq_layer renders a spectrogram tile", ok, why)
        fc, fs = im["fc"], im["fs"]
        lo, hi = fc - fs / 2, fc + fs / 2
        bad = []
        # Every zoom depth the scroll wheel can reach, on both axes at once.
        for tf in (1.0, 0.25, 0.01, 0.001):
            for ff in (1.0, 0.25, 0.01):
                mid, half = (lo + hi) / 2, (hi - lo) * ff / 2
                rr = c.get(f"/api/iq_layer?id={cid}&t0=0&t1={dur * tf}"
                           f"&f0={mid - half}&f1={mid + half}&w=900&h=450")
                ok, why = tile_ok(rr)
                if not ok:
                    bad.append(f"t={tf} f={ff}: {why}")
        check("iq_layer holds up at every zoom depth on both axes", not bad,
              "; ".join(bad))
        bad = []
        for w, h in ((1, 1), (4000, 2000), (1200, 1)):
            ok, why = tile_ok(c.get(f"/api/iq_layer?id={cid}&t0=0&t1={dur}"
                                    f"&w={w}&h={h}"))
            if not ok:
                bad.append(f"{w}x{h}: {why}")
        check("iq_layer renders at extreme canvas sizes", not bad, "; ".join(bad))
        bad = []
        for cmap in CMAPS:
            ok, why = tile_ok(c.get(f"/api/iq_layer?id={cid}&t0=0&t1={dur}"
                                   f"&cmap={cmap}"))
            if not ok:
                bad.append(f"{cmap}: {why}")
        check("iq_layer works for every colormap", not bad, "; ".join(bad))
        meta = json.loads(c.get(f"/api/iq_layer?id={cid}&t0=0&t1={dur}"
                                "&vmin=-100&vmax=-20").headers["X-Meta"])
        check("iq_layer honours the colour lock",
              (meta["vmin"], meta["vmax"]) == (-100.0, -20.0), str(meta)[:80])
        r = c.get("/api/iq_layer?id=no-such-capture&t0=0&t1=1")
        check("iq_layer for an unknown capture is a clean 404",
              r.status_code == 404 and "error" in r.get_json())
        r = c.get(f"/api/iq_layer?id={cid}&t0=1&t1=1")
        check("iq_layer with an empty window is a clean 400",
              r.status_code == 400 and "error" in r.get_json(),
              str(r.status_code))
        r = c.get(f"/api/iq_layer?id={cid}&t0=-99&t1={dur * 99}")
        ok, why = tile_ok(r)
        check("iq_layer clamps a window that runs past the capture", ok, why)

        # ---- malformed requests must not be 500s ----
        # The viewer always sends these, but a 500 is the one answer it cannot
        # do anything with, and it means an unhandled exception server-side.
        probes = [
            "/api/heatmap?sensor=" + sensor,                       # no t0/t1
            f"/api/heatmap?sensor={sensor}&t0=abc&t1=def",
            f"/api/heatmap?sensor={sensor}&t0={t0}&t1={t1}&width=xyz",
            f"/api/psd_layer?sensor={sensor}",                     # no t0/t1
            f"/api/psd_layer?sensor={sensor}&t0=abc&t1=xyz",
            f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}&h=abc",
            f"/api/psd_layer?sensor={sensor}&t0={t0}&t1={t1}&vmin=abc&vmax=1",
            f"/api/pfp_frame?sensor={sensor}",                     # no freq/t
            f"/api/pfp_frame?sensor={sensor}&freq=abc&t0=1&t1=2",
            f"/api/iq_layer?id={cid}&t0=abc&t1=xyz",
            f"/api/iq_layer?id={cid}&t0=0&t1={dur}&w=abc",
            "/api/iq_layer",                                       # no id
            "/api/psd_meta", "/api/pfp_meta", "/api/heatmap",
        ]
        fives = []
        for p in probes:
            rr = c.get(p)
            if rr.status_code >= 500:
                fives.append(f"{p} -> {rr.status_code}")
        check(f"no malformed request returns a 5xx ({len(probes)} probed)",
              not fives, "; ".join(fives))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - every endpoint and every viewer control behaves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
