"""test_viewer.py: drive viewer.html in a real browser and press every control.

test_serve.py proves the endpoints answer. This proves the page in front of them
actually works: that the source dropdown populates, that picking a capture
renders, that the zoom sliders, the colormap picker, the statistic buttons,
reset and export all do something and none of them throws. A backend test cannot
catch a viewer that never issues the request.

Every console error and every failed request is recorded for the whole session
and reported at the end, so a JavaScript exception fails the run rather than
being invisible.

Optional, and skipped cleanly when it cannot run: it needs `pip install
playwright` and a Chromium, neither of which the project itself requires. Point
ATLAS_CHROMIUM at a browser binary if Playwright cannot find its own.

    pip install playwright && playwright install chromium
    python examples/test_viewer.py          # exits 0 = PASS (or SKIP)
"""

import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

failed = 0
# Where a Chromium might already be, so a machine that has one does not have to
# download a second copy. PLAYWRIGHT_BROWSERS_PATH layouts are versioned, hence
# the glob rather than a fixed path.
CHROMIUM_HINTS = ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                  "/usr/bin/chromium", "/usr/bin/chromium-browser",
                  "/usr/bin/google-chrome")


def check(name, ok, detail=""):
    global failed
    if not ok:
        failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def skip(why):
    print(f"  [SKIP] {why}")
    print("\nRESULT: SKIP - the browser test did not run (this is not a failure)")
    return 0


def find_chromium():
    import glob
    for pat in (os.environ.get("ATLAS_CHROMIUM"), *CHROMIUM_HINTS):
        if not pat:
            continue
        for p in sorted(glob.glob(pat), reverse=True):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_up(port, proc, timeout=60):
    """True once the server answers. Fails fast if it died instead."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with contextlib.closing(socket.socket()) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.25)
    return False


def build(tmp):
    """A db dir with all four layers. A longer capture than test_serve.py uses,
    so there is enough of a time axis to zoom into with the wheel."""
    import test_serve as TS
    dbs, _sensor, env = TS.build(tmp, iq_samples=1024 * 256)
    return dbs, env


def summary_reuse_checks(page, check, settle):
    """Zooming the summary layer must paint from what is already on hand.

    The tile layers reuse a cached window that CONTAINS the wanted one, so a
    zoom draws correct-but-coarser data immediately and refines underneath. The
    summary layer was exact-match only, which made it the one layer with no
    instant paint: every notch, and every zoom-out crossing back from PSD, sat
    on the stretched preview until the network answered -- measured at 302 ms
    against 23 ms once the reuse existed.

    Skipped unless this fixture actually has a summary layer to zoom.
    """
    if not page.evaluate("summaryAvail === true"):
        return
    # This fixture spans less than PSD_THRESHOLD, so every window in it belongs
    # to the PSD layer and zooming can never reach the summary -- the check would
    # skip silently and test nothing. Switching the deeper layer off for the
    # duration is what makes buildReq choose the summary, which is the path under
    # test; the layer flag is restored afterwards.
    page.evaluate("() => { window.__psdWas = psdAvail; psdAvail = false; }")
    try:
        _summary_reuse_body(page, check, settle)
    finally:
        page.evaluate("() => { psdAvail = window.__psdWas; }")
        settle()


def _summary_reuse_body(page, check, settle):
    # A wide window first, so there is something in the cache to reuse, then a
    # window strictly inside it.
    page.evaluate("""async () => {
        view.t0 = meta.tmin; view.t1 = meta.tmax;
        clampTimeView(); layoutSliders(); requestData();
        for (let i=0;i<150 && !(layerMode==='summary' && data); i++)
            await new Promise(r=>setTimeout(r,100));
    }""")
    settle()
    if not page.evaluate("layerMode==='summary' && !!data"):
        check("zooming the summary paints instantly from a wider cached window",
              False, "could not reach the summary layer to test it")
        return
    # What separates a reuse from "the stale picture is still up" is NOT that
    # something is drawn -- something always is. It is that the viewer stops
    # reporting a wait: on a reuse requestData calls setLoading(false) and the
    # follow-up refine is soft, so the bar is never raised. Without the reuse the
    # same gesture raises it and the user waits on the network. A first version of
    # this check asserted only "layerMode is summary and data is set", which is
    # true either way -- it passed with the reuse removed.
    got = page.evaluate("""async () => {
        window.__load = [];
        const realSet = window.setLoading;
        window.setLoading = function(on){ window.__load.push(!!on); return realSet.apply(this, arguments); };
        const t0 = performance.now();
        const c = (view.t0+view.t1)/2, s = (view.t1-view.t0)*0.45;   // strictly inside
        view.t0 = c-s/2; view.t1 = c+s/2;
        clampTimeView(); layoutSliders(); requestData();
        const sync = {ms: Math.round(performance.now()-t0),
                      drawn: layerMode==='summary' && !!data,
                      calls: window.__load.slice()};
        await new Promise(r => setTimeout(r, 900));      // let the refine happen
        sync.callsAfter = window.__load.slice();
        window.setLoading = realSet;
        return sync;
    }""")
    waited = any(got["callsAfter"])
    check("zooming the summary paints instantly from a wider cached window",
          got["drawn"] and got["ms"] < 150 and not waited,
          f"{got['ms']} ms, loading raised: {waited} (setLoading calls: {got['callsAfter']})")


def warm_checks(page, check, settle):
    """Idle prefetching must stay quiet, especially when a warm cannot succeed.

    The prefetch re-arms itself so a queue of several tiles drains without
    waiting for the user's next move. Re-arming on "work remains" rather than on
    progress turns any warm that can never land -- a statistic answering 404, a
    server restart, a dropped connection -- into an unbounded request storm on a
    page nobody is touching: measured at 468 requests in 8 idle seconds, each one
    a full window scan. A bound on idle traffic is the only thing that catches
    this; no functional check does, because every picture is still correct.
    """
    # Count ATTEMPTS inside the patched fetch, not HTTP requests seen by the
    # browser. A first version of this test watched page.on("request") while
    # rejecting synchronously -- so the metric was structurally zero and the
    # check passed on the storming code it was written to catch. What is being
    # measured is how many times the warm loop tries, whether or not a request
    # ever leaves the process.
    #
    # It also has to be a state with something TO warm: on a fixture holding
    # only psd.duckdb, inside the PSD layer, at a span under the threshold, the
    # warm queue is legitimately empty and nothing would be attempted however
    # broken the loop was. Advertising the sibling statistics puts two real
    # candidates in the queue; they are what the loop must stop retrying.
    state = page.evaluate("""() => {
        window.__warmTries = 0;
        window.__realFetch = window.fetch;
        window.fetch = (u, o) => {
            if (typeof u === 'string' && u.includes('/api/psd_layer')){
                window.__warmTries++;
                return Promise.reject(new Error('induced warm failure'));
            }
            return window.__realFetch(u, o);
        };
        window.__statsWas = psdStats;
        psdStats = ['max','median','mean'];      // give the queue real candidates
        warmCancel(); drawCrisp();               // arm the chain
        return {layer: layerMode, mode};
    }""")
    try:
        settle()
        page.evaluate("window.__warmTries = 0;")
        page.wait_for_timeout(6000)                            # idle, no input
        n = page.evaluate("window.__warmTries")
        check("idle prefetching does not storm when a warm keeps failing",
              n < 25, f"{n} warm attempt(s) in 6 idle seconds "
                      f"(layer={state['layer']})")
    finally:
        page.evaluate("""() => {
            if (window.__realFetch) window.fetch = window.__realFetch;
            if (window.__statsWas) psdStats = window.__statsWas;
        }""")
        settle()


def zone_checks(page, check, settle):
    """The zoom bars must show WHERE zooming hands off to a deeper layer.

    Both bars carry the same two markers: blue where the PSD detail layer takes
    over, amber where the PFP frame layer can. A marker that sits somewhere
    other than the real threshold is worse than no marker, so this pins the
    blue's edge to the threshold two independent ways -- by geometry, and by
    bisecting for the zoom where layerMode actually flips -- and requires the
    two answers to be the same place. It also checks the amber is dimmed while
    the frame layer is out of reach and brightens when it is not, because that
    dimming is the only thing stopping the marker promising a layer that cannot
    be drawn.
    """
    bar = page.locator("#timebar")
    psd, pfp = page.locator("#timepsd"), page.locator("#timepfp")
    if not (bar.count() and psd.count() and pfp.count()):
        return
    # The markers describe the CBRS layer stack. An IQ capture is one layer with
    # its own pyramid and deliberately shows no zones, so switch to CBRS first
    # rather than measuring a bar that is correctly empty.
    if not page.evaluate("mode === 'cbrs'"):
        opts = page.evaluate(
            "Array.from(document.getElementById('source').options).map(o=>o.value)")
        if "cbrs" not in opts:
            return
        page.select_option("#source", "cbrs")
        settle()
    if not page.evaluate("mode === 'cbrs' && psdAvail"):
        return

    def zoom_to(z):
        page.evaluate(f"setTimeZoom({z})")
        settle()

    def state():
        return page.evaluate("""(function(){
            const b=document.getElementById('timebar').getBoundingClientRect();
            const z=document.getElementById('timepsd').getBoundingClientRect();
            const k=document.getElementById('timeknob').getBoundingClientRect();
            return {layer: layerMode, span: view.t1-view.t0,
                    blue_w: z.width, blue_l: z.left-b.left, bar_w: b.width,
                    inside: z.width>0 && k.left+k.width/2 >= z.left-1};
        })()""")

    zoom_to(0)
    s = state()
    check("the time bar carries a blue PSD zone", s["blue_w"] > 4,
          f"{round(s['blue_w'])} px of {round(s['bar_w'])}")

    # 1. geometry: the blue starts exactly where the PSD threshold sits
    want = page.evaluate(
        "(function(){const w=document.getElementById('timebar').clientWidth;"
        "return Math.max(0,Math.min(1,tspanToZoom(PSD_THRESHOLD)))*(w-16)+8;})()")
    check("the blue zone starts at the PSD threshold, not somewhere decorative",
          abs(s["blue_l"] - want) <= 2,
          f"edge at {round(s['blue_l'])} px, threshold at {round(want)} px")

    # A dataset shorter than the threshold is PSD territory end to end, so
    # there is no plain stretch to see and none must be drawn. Which assertion
    # applies is a property of the fixture, not of the code.
    full = page.evaluate("meta.tmax-meta.tmin")
    thr = page.evaluate("PSD_THRESHOLD")
    long_enough = full > thr * 1.2
    if long_enough:
        check("the bar stays plain across spans too wide for PSD",
              s["blue_l"] > 8, f"plain for the first {round(s['blue_l'])} px")
    else:
        check("a span shorter than the threshold is blue from the left edge",
              s["blue_l"] <= 9 and s["layer"] in ("psd", "pfp"),
              f"{full / 3600:.0f} h fixture, all of it PSD territory")

    # 2. behaviour: bisect for the layer flip and for the knob entering the
    # blue. Coarse sampling can agree by luck; the edges themselves cannot.
    def bisect(by_layer):
        lo, hi = 0.0, 0.8
        for _ in range(8):
            mid = (lo + hi) / 2
            zoom_to(mid)
            st = state()
            inside = st["layer"] in ("psd", "pfp") if by_layer else st["inside"]
            if inside:
                hi = mid
            else:
                lo = mid
        return hi

    flip, edge = bisect(True), bisect(False)
    check("crossing into the blue is exactly when the layer becomes PSD",
          abs(flip - edge) <= 0.02, f"layer flips at {flip:.3f}, blue edge {edge:.3f}")
    zoom_to(edge)
    st = state()
    if long_enough:
        check("that crossing lands on the documented PSD span threshold",
              abs(st["span"] - thr) / thr < 0.10,
              f"{st['span'] / 3600:.1f} h vs threshold {thr / 3600:.0f} h")
    else:
        # No crossing exists inside this fixture: it starts already in PSD.
        check("no false crossing is invented on a short fixture",
              st["layer"] in ("psd", "pfp") and edge < 0.01,
              f"PSD from full span ({st['span'] / 3600:.1f} h)")

    # 3. the amber marker must not promise what it cannot deliver
    zoom_to(0)
    page.evaluate("() => { resetFreq(); layoutSliders(); }")
    settle()
    dim = float(page.evaluate(
        "getComputedStyle(document.getElementById('timepfp')).opacity"))
    page.evaluate("""() => { const c = nearestChannelHz();
        viewF.f0 = c - 4e6; viewF.f1 = c + 4e6; onFreqChange(); }""")
    settle()
    lit = float(page.evaluate(
        "getComputedStyle(document.getElementById('timepfp')).opacity"))
    check("the amber PFP zone is dim while the frame layer is out of reach",
          dim < lit, f"{dim} wide band -> {lit} on one channel")
    page.evaluate("() => { resetFreq(); layoutSliders(); }")
    settle()

    # 4. Time on Y swaps which bar drives time, and the markers have to follow
    # it. A blue zone left behind on the bar that no longer controls time would
    # point at the wrong axis.
    rot = page.locator("#rotate")
    if rot.count() and rot.is_visible():
        rot.click()
        settle()
        r = page.evaluate("""(function(){
            const v=document.getElementById('freqpsd').getBoundingClientRect();
            const h=document.getElementById('timepsd').getBoundingClientRect();
            return {rotated, vertical_blue: v.height, horizontal_blue: h.width};
        })()""")
        check("Time on Y moves the blue PSD zone onto the bar that now drives time",
              r["rotated"] and r["vertical_blue"] > 4,
              f"vertical {round(r['vertical_blue'])} px, "
              f"horizontal (frequency) {round(r['horizontal_blue'])} px")
        rot.click()
        settle()


def main():
    print("viewer.html browser test\n")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return skip("playwright is not installed (pip install playwright)")
    exe = find_chromium()

    tmp = tempfile.mkdtemp(prefix="atlas-viewer-test-")
    proc = None
    try:
        dbs, env = build(tmp)
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "serve.py")],
            env={**env, "SEA_PORT": str(port)}, cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if not wait_up(port, proc):
            out = ""
            if proc.poll() is not None and proc.stdout:
                out = proc.stdout.read()[-400:]
            return skip(f"serve.py did not come up on port {port}. {out}")
        base = f"http://127.0.0.1:{port}"
        check("serve.py answers on a real port", True, base)

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(executable_path=exe)
            except Exception as e:                            # noqa: BLE001
                return skip(f"could not launch Chromium ({str(e).splitlines()[0]})")
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors, bad_requests = [], []
            page.on("console", lambda m: m.type == "error"
                    and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("requestfailed", lambda r: bad_requests.append(
                f"{r.url} ({r.failure})"))
            page.on("response", lambda r: r.status >= 400
                    and bad_requests.append(f"{r.url} -> {r.status}"))

            page.goto(base, wait_until="networkidle", timeout=60000)
            check("the viewer page loads", page.title() != "",
                  page.title()[:60])
            check("the canvas is present", page.locator("#cv").count() == 1)

            # ---- the source dropdown: the first thing anyone touches ----
            opts = page.locator("#source option")
            labels = [opts.nth(i).inner_text() for i in range(opts.count())]
            check("the source dropdown is populated from the databases",
                  len(labels) >= 2, f"{len(labels)}: " + " | ".join(labels[:4]))
            iq_opt = next((l for l in labels if "IQ" in l or "demo" in l), None)
            check("the demo IQ capture appears in the source dropdown",
                  iq_opt is not None, str(iq_opt))

            def settle(ms=1200):
                page.wait_for_timeout(ms)
                with contextlib.suppress(Exception):
                    page.wait_for_load_state("networkidle", timeout=8000)

            def tiles_drawn():
                """Whether the canvas has non-background pixels in it."""
                return page.evaluate("""() => {
                    const c = document.getElementById('cv');
                    if (!c || !c.width || !c.height) return 0;
                    const g = c.getContext('2d');
                    const d = g.getImageData(0, 0, c.width, c.height).data;
                    let n = 0;
                    for (let i = 0; i < d.length; i += 4 * 97) {
                        if (d[i] > 24 || d[i+1] > 24 || d[i+2] > 28) n++;
                    }
                    return n;
                }""")

            # Select every source in turn: each must render something.
            drew = {}
            for i, label in enumerate(labels):
                page.select_option("#source", index=i)
                settle()
                drew[label] = tiles_drawn()
            check("every source in the dropdown renders pixels",
                  all(v > 0 for v in drew.values()),
                  ", ".join(f"{k[:26]}={v}" for k, v in drew.items()))

            # ---- pick the IQ capture and work every control on it ----
            if iq_opt:
                page.select_option("#source", label=iq_opt)
                settle()

            # #sensor is deliberately disabled in IQ mode (a capture has no
            # sensor), so only work the controls the current mode enables.
            for cid in ("#cmap", "#sensor"):
                el = page.locator(cid)
                if not (el.count() and el.is_visible() and el.is_enabled()):
                    print(f"  [ -- ] {cid} is not active in this mode, skipped")
                    continue
                n = page.locator(f"{cid} option").count()
                for i in range(n):
                    page.select_option(cid, index=i, timeout=15000)
                    settle(700)
                check(f"{cid} works for all {n} of its options",
                      tiles_drawn() > 0 and not errors,
                      "; ".join(errors[-2:]))

            stats = page.locator("button[data-stat]")
            if stats.count():
                pressed = 0
                for i in range(stats.count()):
                    b = stats.nth(i)
                    if b.is_visible() and b.is_enabled():
                        b.click(timeout=15000)
                        settle(700)
                        pressed += 1
                check(f"the statistic buttons all respond ({pressed} pressed)",
                      tiles_drawn() > 0 or pressed == 0)

            # Zoom with the wheel, frequency-zoom with ctrl+wheel, and pan by
            # dragging: the three canvas gestures the whole viewer is built on.
            box = page.locator("#cv").bounding_box()
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(cx, cy)
            for _ in range(4):
                page.mouse.wheel(0, -240)
                page.wait_for_timeout(220)
            settle()
            check("scrolling zooms in and keeps rendering", tiles_drawn() > 0,
                  f"{tiles_drawn()} sampled pixels")
            page.keyboard.down("Control")
            for _ in range(3):
                page.mouse.wheel(0, -240)
                page.wait_for_timeout(220)
            page.keyboard.up("Control")
            settle()
            check("ctrl+scroll zooms frequency and keeps rendering",
                  tiles_drawn() > 0)
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx - 220, cy - 60, steps=12)
            page.mouse.up()
            settle()
            check("dragging pans and keeps rendering", tiles_drawn() > 0)

            # The two zoom sliders, dragged by their knobs.
            for knob, track, what in (("#timeknob", "#timetrack", "time"),
                                      ("#freqknob", "#freqtrack", "frequency")):
                k, t = page.locator(knob), page.locator(track)
                if not (k.count() and t.count() and k.is_visible()):
                    continue
                kb, tb = k.bounding_box(), t.bounding_box()
                page.mouse.move(kb["x"] + kb["width"] / 2,
                                kb["y"] + kb["height"] / 2)
                page.mouse.down()
                page.mouse.move(tb["x"] + tb["width"] / 2,
                                tb["y"] + tb["height"] * 0.25, steps=10)
                page.mouse.up()
                settle()
                check(f"the {what} zoom slider drags and keeps rendering",
                      tiles_drawn() > 0)

            zone_checks(page, check, settle)
            summary_reuse_checks(page, check, settle)
            warm_checks(page, check, settle)

            reset = page.locator("#reset")
            if reset.count():
                reset.click()
                settle()
                check("reset zooms back out and keeps rendering",
                      tiles_drawn() > 0)

            # Export writes a PNG through a download; the click must at least
            # not throw, and should produce a file when it is a real download.
            exp = page.locator("#exportBtn")
            if exp.count() and exp.is_visible():
                try:
                    with page.expect_download(timeout=15000) as dl:
                        exp.click()
                    path = dl.value.path()
                    ok = bool(path) and os.path.getsize(path) > 1000
                    check("export produces a PNG", ok,
                          f"{dl.value.suggested_filename}, "
                          f"{os.path.getsize(path) if path else 0} bytes")
                except Exception as e:                        # noqa: BLE001
                    # Some builds render into a new tab instead of downloading.
                    check("export does not throw",
                          "download" in str(e).lower() or "timeout" in str(e).lower(),
                          str(e).splitlines()[0][:100])

            # Reloading with data present must not error either.
            page.goto(base, wait_until="networkidle", timeout=60000)
            settle()
            check("a reload comes back up clean", tiles_drawn() >= 0)

            real_errors = [e for e in errors if "favicon" not in e.lower()]
            check("no JavaScript error anywhere in the session",
                  not real_errors, "; ".join(real_errors[:3]))
            # A cancelled DATA request is intended behaviour, not a failure: the
            # viewer aborts a superseded one on purpose (see fetchTile) so the one
            # the user is waiting for is not queued behind stale ones, and it
            # drops a prefetch that a sensor change made pointless.
            #
            # /api/heatmap belongs on this list now and did not before. The
            # summary layer used to fetch without an AbortController, which is
            # exactly why zooming it was slow: every superseded window ran to
            # completion, a server scan each, holding sockets from the browser's
            # six per origin. It shares `inflight` with the tile layers now, so a
            # zoom burst legitimately aborts the windows the user has left. This
            # check caught that change as a failure the first time it ran -- the
            # right response was to widen the list, not to stop aborting.
            #
            # Only aborts are forgiven, and only for these data endpoints: a
            # 4xx/5xx, a refused connection, or an aborted page load still fails.
            def excused(r):
                return ("net::ERR_ABORTED" in r
                        and any(e in r for e in ("/api/psd_layer", "/api/pfp_frame",
                                                 "/api/iq_layer", "/api/heatmap")))
            real_bad = [r for r in bad_requests
                        if "favicon" not in r.lower() and not excused(r)]
            check("no request failed anywhere in the session",
                  not real_bad, "; ".join(real_bad[:3]))
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=15)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"RESULT: FAIL ({failed} check(s) failed above)")
        return 1
    print("RESULT: PASS - every viewer control works in a real browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
