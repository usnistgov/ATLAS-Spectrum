"""serve.py: Flask backend for the spectrum viewer.

One canvas, three continuous CBRS layers (summary -> PSD -> PFP) plus the
independent IQ-capture mode. For any requested window the server picks the
coarsest stored level that still fills the plot, renders it to a WebP tile
and streams the bytes directly (metadata rides in the X-Meta header).

Storage (built by build_db.py / *_ingest.py, then compact_db.py):
    spectrum.duckdb  summaries; dBm as DOUBLE, or SMALLINT dBm*10 once
                     compact_db.py has run (see _dbm_div -- both are served)
    psd.duckdb       psd_chunk: zlib blobs of 256 consecutive uint8 spectra
    pfp.duckdb       pfp_chunk: zlib blobs of 1024 consecutive uint8 frames
    iq.duckdb        iq_stft: uint8 STFT pyramid per capture (iq_ingest.py)

    cd /path/to/spectrum-viewer
    py serve.py            # http://127.0.0.1:8090

"""

import base64
import gzip
import io
import json
import math
import os
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

# Before the third-party imports: serve.py is the script the ingest tells the
# user to run next ("next: python serve.py"), and it was the one entry point with
# no dependency guard -- a half-populated venv answered with a raw
# ModuleNotFoundError while every ingest script printed an actionable message.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ingest"))
import _require  # noqa: F401,E402

import duckdb                                             # noqa: E402
import numpy as np                                        # noqa: E402
from PIL import Image                                     # noqa: E402
from flask import (Flask, Response, jsonify, make_response, request,  # noqa: E402
                   send_file)

HERE = os.path.dirname(os.path.abspath(__file__))
# Each database path is overridable, like IQ_DB below, so a test run or a
# second dataset can point somewhere else without moving files around.
# ATLAS_DB_DIR moves all four databases at once; the per-database variables
# override it one at a time. atlas.py and every ingest script agree on this.
# abspath: atlas.py's doctor resolves this the same way, and without it a
# relative ATLAS_DB_DIR meant the doctor inspected <cwd>/dir while the server
# read <repo>/dir -- a clean bill of health for a database nobody is serving.
DB_DIR = os.path.abspath(os.environ.get("ATLAS_DB_DIR") or HERE)
DB_PATH = os.environ.get("SPECTRUM_DB") or os.path.join(DB_DIR, "spectrum.duckdb")

# (table, bucket_seconds), finest first. 0 == raw native cadence.
LEVELS = [
    ("raw", 0),
    ("lvl_m10", 600),
    ("lvl_h1", 3600),
    ("lvl_h6", 21600),
    ("lvl_d1", 86400),
]

app = Flask(__name__)


@app.errorhandler(Exception)
def _degrade(exc):
    """Answer a storage or memory failure with 503, never a 500.

    A DuckDB OutOfMemoryException from the tile path reached Flask as a 500, and
    a 500 is the one answer the viewer can do nothing with -- it cannot tell
    "retry a narrower window" from "this build is broken". 503 says try again,
    which is exactly right for pressure that passes. Anything that is not a
    storage or memory failure is re-raised so real bugs still surface loudly in
    the log and in the tests.
    """
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (MemoryError, duckdb.Error)):
        app.logger.warning("degrading to 503: %r", exc)
        return jsonify({"error": "the server is short of memory or cannot read "
                                 "the database right now; try a narrower window"
                        }), 503
    raise exc

# Each request gets its own DuckDB cursor off the shared connection rather than
# taking a lock around the shared one. A single lock meant a cheap request --
# /api/psd_meta, or the small tile the user is actually waiting for -- queued
# behind whatever multi-second scan was already running. Measured on a 180-day
# database: /api/psd_meta answered in 3.0 s while PSD tiles were in flight, and
# eight concurrent tiles completed in perfect single file. Cursors are DuckDB's
# supported way to use one database from several threads.

# The CBRS monitoring databases (spectrum/psd/pfp .duckdb) are OPTIONAL and
# INDEPENDENT: they are multi-GB, not shipped with the repo, and a real download
# does not necessarily contain all three. The published SEA data has days of PSD
# and PFP with no Summaries export at all, so spectrum.duckdb simply cannot be
# built from it -- and the sensor list used to come from that one file, so those
# installs advertised no sensors, the viewer dropped CBRS mode entirely, and
# gigabytes of ingested PSD were unreachable behind "CBRS monitoring disabled".
# Every layer knows its own sensors and time range; ask all three.
def _sensor_rows(path, table, count_col):
    if not os.path.exists(path):
        return []
    try:
        c = duckdb.connect(path, read_only=True)
        try:
            return c.execute(f"SELECT sensor, t_min, t_max, {count_col} FROM {table} "
                             "WHERE t_min IS NOT NULL AND t_max IS NOT NULL "
                             "ORDER BY sensor").fetchall()
        finally:
            c.close()
    except Exception as e:
        print(f"[serve] cannot read {table} from {os.path.basename(path)}: {e}")
        return []


def _cbrs_sensors():
    """Every sensor any CBRS layer can draw, widest time range across them."""
    merged = {}
    for path, table, col in (
            (DB_PATH, "meta", "n"),
            (os.environ.get("PSD_DB") or os.path.join(DB_DIR, "psd.duckdb"),
             "psd_meta", "captures"),
            (os.environ.get("PFP_DB") or os.path.join(DB_DIR, "pfp.duckdb"),
             "pfp_meta", "rows")):
        for name, t0, t1, n in _sensor_rows(path, table, col):
            s = merged.setdefault(name, {"sensor": name, "t_min": t0,
                                         "t_max": t1, "n": 0})
            s["t_min"] = min(s["t_min"], t0)
            s["t_max"] = max(s["t_max"], t1)
            s["n"] += int(n or 0)
    return [merged[k] for k in sorted(merged)]


connection, freqs = None, []
if os.path.exists(DB_PATH):
    # Wrapped, unlike before: this open is at module scope, so a truncated,
    # zero-byte or non-DuckDB spectrum.duckdb killed the process with a raw
    # traceback BEFORE the port was bound -- and took the PSD, PFP and IQ layers
    # down with it, none of which had anything wrong. Every other layer already
    # degrades to "not there" on a bad file; this one now does too, which is
    # exactly the state a missing spectrum.duckdb produces and which is known to
    # serve everything else correctly.
    try:
        connection = duckdb.connect(DB_PATH, read_only=True)
        # Static per-boot metadata (sensor list + channel freqs): computed once so
        # /api/meta never rescans the 47M-row raw table per page load.
        freqs = [r[0] for r in connection.execute(
            "SELECT DISTINCT freq FROM raw ORDER BY freq").fetchall()]
    except Exception as e:                                 # noqa: BLE001
        connection, freqs = None, []
        print(f"[serve] {os.path.basename(DB_PATH)} could not be read ({e}); "
              "serving without the channel-summary layer. The PSD, PFP and IQ "
              "layers are unaffected.")


def _dbm_div():
    """What the summary tables store per dBm: 1 for dBm, 10 for dBm*10.

    spectrum.duckdb exists in two shapes. build_db.py writes mx/md/mn as DOUBLE
    real dBm; compact_db.py rewrites them as SMALLINT dBm*10 (lossless at the
    0.1 dBm the API rounds to anyway). serve.py reads both, so the scale has to
    come off the file instead of being assumed -- and it was assumed twice, in
    opposite directions: /api/heatmap always divided by 10, while _psd_match
    never did. On any one database exactly one of them was therefore wrong by a
    factor of 10, which is what put a -980 dBm PSD legend next to a summary
    reading -98, and made the two layers jump when zoom crossed between them.
    """
    if connection is None:
        return 1.0
    for tbl in ("lvl_h1", "lvl_m10", "lvl_d1", "raw"):
        try:
            r = connection.execute("SELECT data_type FROM duckdb_columns() WHERE "
                            "table_name=? AND column_name='mx'", [tbl]).fetchone()
        except Exception:
            continue
        if r:
            return 10.0 if "INT" in r[0].upper() else 1.0
    return 1.0


DBM_DIV = _dbm_div()
# `summary` tells the viewer whether the top layer exists at all. Without it,
# PSD is the zoomed-out view rather than something to fall back FROM.
_META = {"sensors": _cbrs_sensors(), "freqs": freqs, "summary": connection is not None}
if connection is None:
    what = (f"{len(_META['sensors'])} sensor(s) from the PSD/PFP databases"
            if _META["sensors"] else "no CBRS data")
    print(f"[serve] {os.path.basename(DB_PATH)} not found -- no channel summary "
          f"layer; serving {what}.")
    if not _META["sensors"]:
        print("[serve]   IQ capture mode still works. Run examples/make_sample.py "
              "for a zero-download demo.")


# ---- Colormaps + tile encoding (mirrored in viewer.html) ---------------
# The exact matplotlib 256-entry lookup tables, base64 of raw RGB bytes.
#
# These used to be approximations: inferno as 12 anchor colours linearly
# interpolated, turbo as a 5th-order polynomial fit. Measured against
# matplotlib, inferno was off by up to 34/255 (44 of its 256 entries wrong by
# more than 8) and turbo by up to 251/255 on 235 of 256 entries -- so a tile
# could not be compared against the NASCTN reference plots, which are drawn
# with the real thing. A LUT is 768 bytes; there is no reason to approximate it.
#
# Greys_r (dark = low) is included because the SEA day plots draw PSD in it, and
# viridis/plasma/magma because the published figures use those for the summary
# and PFP panels. `fire` is colorcet CET-L4 and `hot` is matplotlib's: both are
# monotonic in lightness with no purple anywhere, which is what a hot-spectrogram
# ramp is usually wanted for -- inferno spends its lower half in purple, and
# turbo peaks in brightness at index 156 and then DARKENS to (122,4,3), so the
# strongest signals come out dimmer than mid ones. Both of those are faithful to
# the originals; they are just poor choices for this, so better ones are offered.
CMAP_B64 = {
    "viridis":
        "RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBx"
        "SCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2E"
        "Qz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaM"
        "OFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6O"
        "Lm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOO"
        "JYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mK"
        "H5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6A"
        "Ka9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJs"
        "TsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NO"
        "gdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4o"
        "vd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg"
        "+OYh++cj/ecl"
        ,
    "fire":
        "AAAABgAADQAAEgAAFgAAGQAAHAAAHwAAIgAAJAAAJgAAKAAAKwAALQAALgAAMAAAMgAANAAANQAANwAAOAAAOgAAOwAA"
        "PQAAPgAAQAAAQQAAQwAARAAARgAARwAASQAASgAATAAATQAATwAAUAAAUgAAUwAAVQAAVgAAWAAAWQEAWwEAXQEAXgEA"
        "YAEAYQEAYwEAZQEAZgEAaAEAaQEAawEAbQEAbgEAcAEAcQEAcwEAdQEAdgEAeAIAegIAewIAfQIAfwIAgAIAggIAhAIA"
        "hQIAhwIAiQIAigIAjAMAjgMAkAMAkQMAkwMAlQMAlgMAmAMAmgMAnAMAnQQAnwQAoQQAogQApAQApgQAqAQAqQQAqwUA"
        "rQUArwUAsAUAsgUAtAUAtgYAuAYAuQYAuwYAvQYAvwcAwAcAwgcAxAcAxggAyAgAyQgAywgAzQkAzwkA0QkA0goA1AoA"
        "1goA2AsA2gsA2wwA3QwA3w0A4Q0A4w4A5A8A5g8A6BAA6hEA6xMA7RQA7hYA8BgA8RsA8h0A8yAA9SMA9iYA9ikA9ywA"
        "+C8A+TIA+TUA+jgA+jsA+z0A+0AA+0MA/EYA/EkA/EsA/U4A/VEA/VMA/VYA/VgA/lsA/l0A/l8A/mIA/mQA/mYA/mgA"
        "/msA/m0A/m8A/nEA/nMA/nUA/ncA/nkA/nwA/34A/4AA/4IA/4MA/4UA/4cA/4kA/4sA/40A/48A/5EA/5MA/5QA/5YA"
        "/5gA/5oA/5wA/50A/58A/6EA/6MA/6QB/6YB/6gB/6oB/6sB/60B/68B/7AB/7IC/7QC/7UC/7cC/7kC/7oC/7wD/70D"
        "/78D/8ED/8IE/8QE/8YE/8cE/8kF/8oF/8wF/84G/88G/9EG/9IH/9QH/9UI/9cI/9kJ/9oJ/9wK/90K/98L/+AL/+IM"
        "/+MN/+UO/+YP/+gQ/+oR/+sS/+0U/+4X//Aa//Ee//Mk//Qq//Uy//c7//hH//lT//ti//ty//yD//2V//6o//66//7M"
        "//7e//7u////"
        ,
    "hot":
        "CwAADQAAEAAAEgAAFQAAGAAAGgAAHQAAIAAAIgAAJQAAJwAAKgAALQAALwAAMgAANQAANwAAOgAAPAAAPwAAQgAARAAA"
        "RwAASgAATAAATwAAUQAAVAAAVwAAWQAAXAAAXwAAYQAAZAAAZgAAaQAAbAAAbgAAcQAAdAAAdgAAeQAAewAAfgAAgQAA"
        "gwAAhgAAiQAAiwAAjgAAkAAAkwAAlgAAmAAAmwAAngAAoAAAowAApQAAqAAAqwAArQAAsAAAswAAtQAAuAAAugAAvQAA"
        "wAAAwgAAxQAAyAAAygAAzQAAzwAA0gAA1QAA1wAA2gAA3QAA3wAA4gAA5AAA5wAA6gAA7AAA7wAA8gAA9AAA9wAA+QAA"
        "/AAA/wAA/wIA/wUA/wgA/woA/w0A/xAA/xIA/xUA/xcA/xoA/x0A/x8A/yIA/yUA/ycA/yoA/ywA/y8A/zIA/zQA/zcA"
        "/zoA/zwA/z8A/0EA/0QA/0cA/0kA/0wA/08A/1EA/1QA/1YA/1kA/1wA/14A/2EA/2QA/2YA/2kA/2sA/24A/3EA/3MA"
        "/3YA/3kA/3sA/34A/4AA/4MA/4YA/4gA/4sA/44A/5AA/5MA/5UA/5gA/5sA/50A/6AA/6IA/6UA/6gA/6oA/60A/7AA"
        "/7IA/7UA/7cA/7oA/70A/78A/8IA/8UA/8cA/8oA/8wA/88A/9IA/9QA/9cA/9oA/9wA/98A/+EA/+QA/+cA/+kA/+wA"
        "/+8A//EA//QA//YA//kA//wA//4A//8D//8H//8L//8P//8T//8X//8b//8f//8i//8m//8q//8u//8y//82//86//8+"
        "//9C//9G//9K//9O//9S//9W//9a//9e//9h//9l//9p//9t//9x//91//95//99//+B//+F//+J//+N//+R//+V//+Z"
        "//+d//+g//+k//+o//+s//+w//+0//+4//+8///A///E///I///M///Q///U///Y///c///f///j///n///r///v///z"
        "///3///7////"
        ,
    "inferno":
        "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoy"
        "FAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAli"
        "OQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJu"
        "XxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBs"
        "hCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1g"
        "qC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9L"
        "ykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox"
        "5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U"
        "9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN"
        "/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD"
        "9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua"
        "+fyd+v2h/P+k"
        ,
    "plasma":
        "DQiHEAeIEweJFgeKGQaMGwaNHQaOIAaPIgaQJAaRJgWRKAWSKgWTLAWULgWVLwWWMQWXMwWXNQSYNwSZOASaOgSaPASb"
        "PgScPwScQQSdQwOeRAOeRgOfSAOfSQOgSwOhTAKhTgKiUAKiUQKjUwKjVQKkVgGkWAGkWQGlWwGlXAGmXgGmYAGmYQCn"
        "YwCnZACnZgCnZwCoaQCoagCobACobgCobwCocQCocgGodAGodQGodwGoeAGoegKoewKofQOofgOogASogQSngwWnhAWn"
        "hgamhwemiAimigmliwqljQuljgykjw2kkQ6jkg+jlBCilRGhlhOhmBSgmRWfmhafnBeenRidnhmdoBqcoRuboh2aox6a"
        "pR+ZpiCYpyGXqCKWqiOVqySUrCaUrSeTriiSsCmRsSqQsiuPsyyOtC6NtS+MtjCLtzGKuDKJujOIuzSIvDWHvTeGvjiF"
        "vzmEwDqDwTuCwjyBwz2AxD5/xUB+xkF9x0J8yEN7yUR6ykV6y0Z5zEd4zEl3zUp2zkt1z0x00E1z0U5y0k9x01Fx1FJw"
        "1VNv1VRu1lVt11Zs2Fdr2Vhq2lpq2ltp21xo3F1n3V5m3l9l3mFk32Jj4GNj4WRi4mVh4mZg42hf5Gle5Wpd5Wtd5mxc"
        "525b529a6HBZ6XFY6XJX6nRX63VW63ZV7HdU7XlT7XpS7ntR73xR735Q8H9P8IBO8YFN8YNM8oRL84VL84dK9IhJ9IlI"
        "9YtH9YxG9o1F9o9E95BE95FD95NC+JRB+JVA+Zc/+Zg++Zo++ps9+pw8+p47+586+6E5+6I4/KM4/KU3/KY2/Kg1/Kk0"
        "/asz/awz/a4y/a8x/bEw/bIv/bQv/bUu/rct/rgs/ros/rsr/r0q/r4q/sAp/cIp/cMo/cUn/cYn/cgn/com/csm/M0l"
        "/M4l/NAl/NIl+9Mk+9Uk+9ck+tgk+tok+dwk+d0l+N8l+OEl9+Il9+Ql9uYm9ugm9ekm9esn9O0n8+4n8/An8vIn8fQm"
        "8fUl8Pck8Pkh"
        ,
    "magma":
        "AAAEAQAFAQEGAQEIAgEJAgILAgINAwMPAwMSBAQUBQQWBgUYBgUaBwYcCAceCQcgCggiCwkkDAkmDQopDgsrEAstEQwv"
        "Eg0xEw00FA42FQ44Fg87GA89GRA/GhBCHBBEHRFHHhFJIBFLIRFOIhFQJBJTJRJVJxJYKRFaKhFcLBFfLRFhLxFjMRFl"
        "MxBnNBBpNhBrOBBsOQ9uOw9wPQ9xPw9yQA90Qg91RA92RRB3RxB4SRB4ShB5TBF6ThF7TxJ7URJ8UhN8VBN9VhR9VxV+"
        "WRV+WhZ+XBZ/XRd/Xxh/YBiAYhmAZBqAZRqAZxuAaByBahyBax2BbR2Bbh6BcB+Bch+BcyCBdSGBdiGBeCKBeSKCeyOC"
        "fCOCfiSCgCWCgSWBgyaBhCaBhieBiCeBiSiBiymBjCmBjiqBkCqBkSuBkyuAlCyAliyAmC2AmS2Amy5/nC5/ni9/oC9/"
        "oTB+ozB+pTF+pjF9qDJ9qjN9qzN8rTR8rjR7sDV7sjV7szZ6tTZ6tzd5uDd5ujh4vDl4vTl3vzp3wDp2wjt1xDx1xTx0"
        "xz1zyD5zyj5yzD9xzUBxz0Bw0EFv0kJv00Nu1URt1kVs2EVs2UZr20dq3Ehp3klo30po4Exn4k1m405l5E9k5VBk51Jj"
        "6FNi6VRi6lZh61dg7Fhg7Vpf7lte711e8F9e8WBd8mJd8mRc82Vc9Gdc9Glc9Wtc9mxc9m5c93Bc93Jc+HRc+HZc+Xhd"
        "+Xld+Xtd+n1e+n9e+oFf+4Nf+4Vg+4dh/Ilh/Ipi/Ixj/I5k/JBl/ZJm/ZRn/ZZo/Zhp/Zpq/Ztr/p1s/p9t/qFu/qNv"
        "/qVx/qdy/qlz/qp0/qx2/q53/rB4/rJ6/rR7/rZ8/rd+/rl//ruB/r2C/r+E/sGF/sKH/sSI/saK/siM/sqN/syP/s2Q"
        "/s+S/tGU/tOV/tWX/teZ/tia/dqc/dye/d6g/eCh/eKj/eOl/eWn/eep/emq/eus/Oyu/O6w/PCy/PK0/PS2/Pa4/Pe5"
        "/Pm7/Pu9/P2/"
        ,
    "turbo":
        "MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/"
        "RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9"
        "RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXi"
        "IMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeyn"
        "J+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5c"
        "ef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0"
        "vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s86"
        "78068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks"
        "/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ"
        "7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYD"
        "wyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcC"
        "gQYCfgUCegQD"
        ,
    "greys":
        "AAAAAQEBAgICAwMDBQUFBgYGBwcHCAgICQkJCgoKDAwMDQ0NDg4ODw8PEBAQERERExMTFBQUFRUVFhYWFxcXGBgYGhoa"
        "GxsbHBwcHR0dHh4eHx8fISEhIiIiIyMjJCQkJSUlJycnKCgoKSkpKysrLCwsLi4uLy8vMDAwMjIyMzMzNTU1NjY2ODg4"
        "OTk5Ojo6PDw8PT09Pz8/QEBAQUFBQ0NDRERERkZGR0dHSEhISkpKS0tLTU1NTk5OUFBQUVFRUlJSU1NTVFRUVVVVVlZW"
        "V1dXWFhYWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5u"
        "b29vcHBwcXFxcnJyc3NzdXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gYGBgoKCg4ODhISEhYWFhoaGh4eH"
        "iIiIiYmJioqKjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVl5eXmJiYmZmZmpqanJycnZ2dnp6en5+foKCgoqKi"
        "o6OjpKSkpaWlp6enqKioqampqqqqq6urra2trq6ur6+vsLCwsrKys7OztLS0tbW1tra2uLi4ubm5urq6u7u7vb29vr6+"
        "vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS"
        "09PT1NTU1NTU1dXV1tbW19fX2NjY2dnZ2tra2tra29vb3Nzc3Nzc3d3d3t7e39/f39/f4ODg4eHh4eHh4uLi4+Pj5OTk"
        "5OTk5eXl5ubm5+fn5+fn6Ojo6enp6enp6urq6+vr7Ozs7Ozs7e3t7u7u7u7u7+/v8PDw8PDw8fHx8fHx8vLy8vLy8/Pz"
        "8/Pz9PT09PT09fX19fX19vb29vb29/f39/f39/f3+Pj4+Pj4+fn5+fn5+vr6+vr6+/v7+/v7/Pz8/Pz8/f39/f39/v7+"
        "/v7+////////"
        ,
}


def _load_luts():
    return {k: np.frombuffer(base64.b64decode(v), np.uint8).reshape(256, 3)
            for k, v in CMAP_B64.items()}


LUTS = _load_luts()
DEFAULT_CMAP = "viridis"


class BadRequest(Exception):
    """A request that cannot be honoured as asked. Answered with 400."""


@app.errorhandler(BadRequest)
def _bad_request(e):
    return jsonify({"error": str(e)}), 400


def _num(name, default=None, cast=float, required=False):
    """One numeric query parameter, parsed.

    Every axis value arrives as text in a URL, so a missing or non-numeric one
    is the caller's mistake and deserves a 400 naming the parameter. Calling
    float() on it directly meant an unhandled exception and a 500 -- the one
    answer the viewer can do nothing with, and indistinguishable from the
    server being broken.
    """
    raw = request.args.get(name)
    if raw is None or raw == "":
        if required:
            raise BadRequest(f"missing required parameter '{name}'")
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"parameter '{name}' must be a number, got {raw!r}")


def _window(default_h=600):
    """The (t0, t1, H) every layer endpoint starts from."""
    return (_num("t0", required=True), _num("t1", required=True),
            max(1, _num("h", 600, int)))


def _locked_scale():
    """The colour lock, when the viewer sent both halves of it, else None."""
    lo, hi = _num("vmin"), _num("vmax")
    return None if lo is None or hi is None else (lo, hi)








def _colorize(mat, vmin, vmax, cmap=DEFAULT_CMAP):
    lut = LUTS.get(cmap, LUTS[DEFAULT_CMAP])
    idx = np.clip((mat - vmin) / max(vmax - vmin, 1e-6) * 255, 0, 255)
    nan = ~np.isfinite(mat)
    idx[nan] = 0
    rgb = lut[idx.astype(np.uint8)]
    # Exactly the client's plot background (#05070b). It was (8,9,12) -- close,
    # but a tile's no-data areas rendered a visibly different shade from the
    # client-drawn summary's, so every layer switch re-tinted the gaps.
    rgb[nan] = (5, 7, 11)
    return rgb


def _tile(rgb, meta):
    """WebP tile bytes + metadata in the X-Meta header (no base64/JSON body:
    ~25% fewer bytes and the browser decodes the image natively)."""
    buf = io.BytesIO()
    # Encoding the tile, not reading the database, is what a zoomed-in request
    # spends its time on: a 5-minute window holding two captures still has to
    # encode the full 2400x560 image, and at method=4 that was 94 ms of a 151 ms
    # request. method=2 produces a byte-for-byte identical 14.1 KB on real frame
    # tiles in 32 ms -- the extra effort levels above 2 buy nothing here because
    # a spectrogram has little for the predictor to exploit. Measured on HU.
    Image.fromarray(rgb, "RGB").save(buf, format="WEBP", quality=80, method=2)
    resp = Response(buf.getvalue(), mimetype="image/webp")
    resp.headers["X-Meta"] = json.dumps(meta)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


def _gz(resp):
    """gzip a JSON response when the client accepts it (summary payloads are
    a few hundred KB of number arrays -> ~4x smaller)."""
    if "gzip" in request.headers.get("Accept-Encoding", "") and resp.status_code == 200:
        resp.set_data(gzip.compress(resp.get_data(), 5))
        resp.headers["Content-Encoding"] = "gzip"
    return resp


def pick_level(span_seconds, target_cols):
    """Finest level whose bucket count over this span <= target_cols."""
    for tbl, bucket in LEVELS:
        if bucket == 0:
            if span_seconds <= 6 * 3600:   # raw only when the window is cheap
                return tbl, bucket
            continue
        if span_seconds / bucket <= target_cols:
            return tbl, bucket
    return LEVELS[-1]


# ---- Time binning, shared by the PSD and PFP layers --------------------
# Both layers used to return one image column per stored capture, which made
# the cost of a tile scale with the window rather than with the screen, and put
# captures at index positions rather than at their real times (so a gap slid
# everything after it sideways). Both now bin onto a uniform grid over the
# requested window: bounded work, and X is honestly time.
COLS_PER_PX = 2           # tile columns per screen pixel (matches iq_layer)
MAX_TILE_COLS = 4096
# Safety valve: at most this many stored captures are read per output column.
# A column is one screen pixel's worth of time, and its value is a max, so the
# tenth-widest capture in a column changes nothing you can see -- but reading
# all 72 captures behind a column of a six-month window costs seconds. Beyond
# this the source is strided, which is what keeps a year of a multi-GB database
# as cheap to draw as a day of it. Zoom in and the stride falls back to 1, so
# the fidelity is only ever traded away at zoom levels that cannot show it.
CAPTURES_PER_COL = 10
READ_BATCH = 16           # chunk rows pulled from DuckDB at a time
# zlib.decompress releases the GIL, so inflating a batch of chunks in parallel
# is a near-linear win on the wide windows where decompression dominates
# (measured: full-span PSD tile 3.3 s -> ~1 s on 4 cores). Results come back
# in submission order, so tiles are byte-identical to the sequential path.
_INFLATE = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
# Decompressed-chunk LRU: pan/zoom windows overlap heavily, so the same chunks
# are inflated over and over -- and inflation is the whole cost of a tile scan.
# ~0.6 MB per entry decompressed; 192 entries bounds this near 110 MB.
_CHUNKS, _chunks_lock, _CHUNKS_MAX = {}, threading.Lock(), 192


def _inflate_cached(key, tblob, blob, n, nf):
    with _chunks_lock:
        hit = _CHUNKS.pop(key, None)
        if hit is not None:
            _CHUNKS[key] = hit          # LRU touch
            return hit
    val = (np.frombuffer(zlib.decompress(tblob), np.float64),
           np.frombuffer(zlib.decompress(blob), np.uint8).reshape(n, nf))
    with _chunks_lock:
        _CHUNKS[key] = val
        while len(_CHUNKS) > _CHUNKS_MAX:
            _CHUNKS.pop(next(iter(_CHUNKS)))
    return val
# Captures per batch on the uncompacted schema (READ_BATCH * this). Measured on
# 172,800 captures: 1024 takes 0.85 s, 16384 takes 2.8 s -- the cost is the
# reduceat over each batch, so a batch far bigger than a screenful of columns
# is pure waste. Bigger is NOT better here.
PSD_ROWS_PER_BATCH = 64
# Colour-scale sampling: this many short reads spread across a sensor's range.
SCALE_PROBES, SCALE_PER_PROBE = 16, 16


def _grid_cols(w):
    """Tile columns for a window drawn `w` screen pixels wide."""
    return max(1, min(int(w) * COLS_PER_PX, MAX_TILE_COLS))


def _stride_for(n_captures, cols):
    """Take every Nth source row when a window holds more than we will read."""
    budget = max(1, cols) * CAPTURES_PER_COL
    if n_captures <= budget:
        return 1
    return int(math.ceil(n_captures / budget))


class _Grid:
    """Max-pool captures onto `cols` uniform time bins across [t0,t1).

    Fed one batch at a time so peak memory is one batch, not one window: a
    six-month PSD window costs the same resident bytes as a six-hour one.
    Bins that no capture landed in stay empty and are rendered as background,
    the same way the summary layer leaves real gaps visible.
    """

    def __init__(self, t0, t1, cols, nbins):
        self.t0, self.span, self.cols = t0, max(t1 - t0, 1e-9), cols
        self.buf = np.zeros((cols, nbins), np.uint8)
        self.filled = np.zeros(cols, bool)
        self.ncap = 0

    def add(self, ts, mat):
        """ts: capture times (ascending). mat: (len(ts), nbins) uint8."""
        if not len(ts):
            return
        self.ncap += len(ts)
        idx = np.clip(((ts - self.t0) / self.span * self.cols).astype(np.int64),
                      0, self.cols - 1)
        # ts is ascending, so equal indices are contiguous: reduceat gives the
        # per-bin max in one pass without a Python loop over captures.
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        red = np.maximum.reduceat(mat, starts, axis=0)
        col = idx[starts]                       # unique within this batch
        self.buf[col] = np.maximum(self.buf[col], red)
        self.filled[col] = True

    def image(self, qmin, qmax, h, offset=0.0):
        """(rows, cols) float32 in dBm, row 0 = low f, empty columns NaN.

        The second axis is pooled down to at most `h` rows first, so a tile is
        never taller than the plot it is drawn into.
        """
        b = self.buf                                   # (cols, nbins)
        nb = b.shape[1]
        # bins_per_row: how many stored frequency bins pool into one pixel row
        bins_per_row = max(1, math.ceil(nb / max(1, h)))
        fb = bins_per_row
        if fb > 1:
            pad = (-nb) % fb
            if pad:                                    # pad < fb, so no group
                b = np.pad(b, ((0, 0), (0, pad)))      # is padding-only
            # reshape(...).max(axis=2) is the obvious way to pool, but is ~10x
            # slower than either alternative below on this uint8 shape -- numpy's
            # generic reduce over a newly-introduced small axis does not vectorize
            # well here. fb is almost always small (NF=2250 bins pooled onto a few
            # hundred pixel rows), where a handful of strided elementwise max()
            # calls beats reduceat; fb only gets large when h itself is tiny (a
            # sliver of a plot), where reduceat wins instead. Both are exact,
            # verified byte-for-byte against the reshape form on real data.
            if fb <= 16:
                pooled = b[:, 0::fb]
                for k in range(1, fb):
                    pooled = np.maximum(pooled, b[:, k::fb])
                b = pooled
            else:
                b = np.maximum.reduceat(b, np.arange(0, b.shape[1], fb), axis=1)
        img = qmin + (b.T.astype(np.float32) / 255.0) * (qmax - qmin) + offset
        if self.filled.any():
            # Sample-and-hold: stretch every capture until the next one, so a
            # zoomed-in window renders continuously instead of as 1-px lines at
            # each capture's exact instant. This finishes what _psd_window's
            # "data gap -> hold the nearest capture" already does for windows
            # with no captures at all.
            idx = np.where(self.filled, np.arange(self.cols), -1)
            idx = np.maximum.accumulate(idx)           # previous filled column
            idx[idx < 0] = int(np.argmax(self.filled))  # backfill leading edge
            img = img[:, idx]
        else:
            img[:, :] = np.nan                         # nothing -> background
        return img


@app.route("/")
def index():
    resp = make_response(send_file(os.path.join(HERE, "viewer.html")))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/meta")
def meta():
    return jsonify(_META)


@app.route("/api/heatmap")
def heatmap():
    if connection is None:                     # no CBRS database present (IQ-only install)
        return _gz(jsonify({"level": None, "bucket": 0, "count": 0,
                            "freq": [], "t": [], "max": [], "median": [], "mean": []}))
    sensor = request.args.get("sensor")
    t0 = _num("t0", required=True)
    t1 = _num("t1", required=True)
    width = max(1, _num("width", 1600, int))
    span = max(t1 - t0, 1.0)

    tbl, bucket = pick_level(span, width)
    rows = connection.cursor().execute(f"""
        SELECT freq, t, mx, md, mn
        FROM {tbl}
        WHERE sensor = ? AND t >= ? AND t < ?
        ORDER BY t
    """, [sensor, t0, t1]).fetchall()

    # Columnar payload, in real dBm. DBM_DIV is 10 on a compacted database (dBm
    # stored as SMALLINT dBm*10) and 1 on the shape the ingest writes -- read off
    # the file, not assumed; see _dbm_div. (The mx/md/mn COLUMN names are the
    # on-disk schema and stay as they are; these local names spell them out.)
    frequencies, times = [], []
    max_values, median_values, mean_values = [], [], []
    for row in rows:
        frequencies.append(row[0]); times.append(int(row[1]))
        max_values.append(None if row[2] is None else row[2] / DBM_DIV)
        median_values.append(None if row[3] is None else row[3] / DBM_DIV)
        mean_values.append(None if row[4] is None else row[4] / DBM_DIV)
    return _gz(jsonify({
        "level": tbl, "bucket": bucket, "count": len(rows),
        "freq": frequencies, "t": times,
        "max": max_values, "median": median_values, "mean": mean_values,
    }))


# ---- PSD continuous layer (chunked uint8 spectra, zlib) ----------------
PSD_DB = os.environ.get("PSD_DB") or os.path.join(DB_DIR, "psd.duckdb")
F0 = 3530040000.0     # first PSD bin (Hz)
DF = 80000.0          # bin spacing (Hz)
NF = 2250             # bins  (full band 3530.04 .. 3709.96 MHz)
# PSD is dBm/Hz; summaries are channel power (dBm). +10*log10(10 MHz) puts the
# PSD layer on the SAME dBm scale so colours line up across the zoom boundary.
PSD_DBM_OFFSET = 70.0

_psd_connection = None
_psd_scale_cache = {}   # sensor -> (vmin, vmax) sampled across the whole range
_psd_kind = None        # 'chunk' | 'rows' | None
# Coarse-time PSD levels (bucket seconds, finest first) if ingest/build_psd_levels.py
# has been run. Empty = no pyramid; every window is read from the captures, which
# is what always happened and is why a very wide one took many seconds.
_psd_lvls = []
_psd_init = threading.Lock()


def _rows(c, table):
    """How many rows a table holds; 0 when it cannot be counted."""
    try:
        return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] or 0
    except Exception:                                      # noqa: BLE001
        return 0


def table_names(c):
    try:
        return {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
    except Exception:
        return set()


# The PSD layer exists per STATISTIC: `max` lives in psd.duckdb (the original
# path, untouched), `median` and `mean` in sibling files psd_median.duckdb /
# psd_mean.duckdb written by `psd_ingest.py --stat median|mean`. Separate files
# rather than a schema change on purpose: every existing database keeps working
# unmodified, and a stat the user never ingested simply is not there.
PSD_LAYER_STATS = ("max", "median", "mean")
_psd_dbs = {}           # stat -> {"connection", "kind", "levels"} once opened
_psd_retry = {}         # stat -> monotonic time before which not to retry a
                        # failed open (an ingest holding the write lock)
OPEN_RETRY = 20.0


def psd_db_path(stat):
    if stat == "max":
        return PSD_DB
    base = os.path.dirname(PSD_DB) or "."
    return os.path.join(base, f"psd_{stat}.duckdb")


def psd_connection(stat="max"):
    """Persistent read-only handle for one statistic's PSD database (saves the
    ~15ms reopen per zoom).

    Read-only, so serving can never damage a database. It does NOT mean the file
    is immutable while the server runs: psd_ingest.py appends to it in place and
    build_psd_levels.py writes psd_lvl into it, which is why both tell you to
    stop serve.py first. Only compact_db.py builds a new file and swaps -- and
    this handle keeps the old inode after that swap, so a compaction under a live
    server means restarting it to see the result.
    """
    global _psd_connection, _psd_kind, _psd_lvls
    entry = _psd_dbs.get(stat)
    if entry is not None:
        return entry["connection"]
    path = psd_db_path(stat)
    if stat not in PSD_LAYER_STATS or not os.path.exists(path):
        return None
    until = _psd_retry.get(stat)
    if until is not None and time.monotonic() < until:
        return None                 # a recent open failed; do not hammer it
    with _psd_init:                 # two requests can race here on first load
        entry = _psd_dbs.get(stat)
        if entry is not None:
            return entry["connection"]
        try:
            connection = duckdb.connect(path, read_only=True)
        except Exception as e:
            # Silence here meant the layer simply was not there: /api/psd_meta
            # answers {"has": false}, the viewer switches the layer off, and
            # nothing anywhere says why. The usual causes are an ingest or
            # compact_db still holding the write lock, and a file that is not a
            # DuckDB database -- both worth naming.
            print(f"[serve] cannot open {os.path.basename(path)}: {e}")
            print(f"[serve]   -- the PSD {stat} layer will not be available "
                  f"(retrying in {OPEN_RETRY:.0f}s).")
            # NOT cached permanently. The usual cause is an ingest or compaction
            # holding the write lock for a few minutes, and caching the failure
            # meant a server that merely STARTED at the wrong moment served
            # {"has": false} for the rest of its life -- the databases were fine
            # and the viewer was told there was no data. Retried after a
            # cooldown, so a genuinely broken file still costs one open per
            # cooldown rather than one per request.
            _psd_retry[stat] = time.monotonic() + OPEN_RETRY
            return None
        # Two on-disk shapes are valid: the compact chunk schema written by
        # compact_db.py, and the row-per-capture schema psd_ingest.py writes.
        # Reading both means compaction is a size optimisation, not a step you
        # can forget and end up with a viewer that renders nothing.
        t = table_names(connection)
        # By row COUNT, not by table presence. chunk_io.schema_of has always
        # decided it this way and its docstring claims to mirror this line; it
        # did not. Two states show the difference. An EMPTY psd_chunk beside a
        # full psd row table advertised the layer and then 404'd every tile. And
        # a database holding data in BOTH -- which a re-ingest after compaction
        # or a hand-merged file produces -- served only the chunked part as if it
        # were everything: measured 6 of 818 captures behind the picture, gap
        # false, 23.7 dB out, with nothing anywhere saying so.
        kind = None
        if "psd_chunk" in t and _rows(connection, "psd_chunk"):
            kind = "chunk"
            if "psd" in t and _rows(connection, "psd"):
                print(f"[serve] WARNING {os.path.basename(path)} holds data in "
                      f"BOTH the psd_chunk and psd shapes. Serving psd_chunk; "
                      f"the row table's captures are NOT drawn. Run "
                      f"ingest/compact_db.py, which refuses this state and "
                      f"explains it.")
        elif "psd" in t and _rows(connection, "psd"):
            kind = "rows"
        levels = []
        if "psd_lvl" in t:
            try:
                levels = sorted(r[0] for r in connection.execute(
                    "SELECT DISTINCT bucket FROM psd_lvl").fetchall())
            except Exception:
                levels = []
        # entry BEFORE the mirror globals: the fast path above returns as soon
        # as the entry exists without taking the lock, so the dict value must
        # already be complete when it becomes visible.
        _psd_dbs[stat] = {"connection": connection, "kind": kind, "levels": levels}
        _scale_stat_budgets()      # the stat-keyed caches share their slots
        if stat == "max":
            # Mirrors kept for the tests and tooling that read these directly;
            # they always describe the max database, the one that always existed.
            _psd_kind, _psd_connection, _psd_lvls = kind, connection, levels
        if kind == "rows":
            print(f"[serve] {os.path.basename(path)} is the uncompacted row "
                  "schema; serving it directly. Run ingest/compact_db.py to shrink it.")
        if levels:
            print(f"[serve] PSD {stat} coarse levels available: "
                  + ", ".join(f"{b}s" for b in levels))
        elif stat == "max":
            print("[serve] no PSD coarse levels; very wide PSD windows will be "
                  "slow. Build them with ingest/build_psd_levels.py.")
    return connection


def psd_stat_info(stat):
    """(connection, kind, levels) for one statistic, opening it if needed."""
    connection = psd_connection(stat)
    entry = _psd_dbs.get(stat) or {"kind": None, "levels": []}
    return connection, entry.get("kind"), entry.get("levels", [])


def psd_stats_available(sensor=None):
    """Which statistics have a database on disk (and this sensor in it)."""
    out = []
    for stat in PSD_LAYER_STATS:
        connection = psd_connection(stat)
        if connection is None:
            continue
        if sensor is not None:
            try:
                # A null time range means the ingest for this statistic read no
                # files: the row exists, the data does not. Offering it lit up a
                # SHOW button whose tiles come back empty, so require a real
                # range -- the same test /api/psd_meta applies before it
                # advertises the layer at all.
                row = connection.cursor().execute(
                    "SELECT 1 FROM psd_meta WHERE sensor=? "
                    "AND t_min IS NOT NULL AND t_max IS NOT NULL LIMIT 1",
                    [sensor]).fetchone()
            except Exception:
                row = None
            if row is None:
                continue
        out.append(stat)
    return out


def _unpack(rows, width):
    """(times, mat) from chunk rows of (n, times_blob, data_blob)."""
    ts = np.concatenate([np.frombuffer(zlib.decompress(r[1]), np.float64) for r in rows])
    mat = np.concatenate([np.frombuffer(zlib.decompress(r[2]), np.uint8).reshape(r[0], width)
                          for r in rows])
    return ts, mat


def _unpack_rows(rows, width):
    """(times, mat) from row-per-capture rows of (t, blob)."""
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    mat = np.stack([np.frombuffer(r[1], np.uint8) for r in rows]) if rows \
        else np.zeros((0, width), np.uint8)
    return ts, mat.reshape(len(rows), width)


def _psd_scale(c, sensor, qmin, qmax, stat="max", kind=None):
    sc = _psd_scale_cache.get((stat, sensor))
    if sc is not None:
        return sc
    full = _psd_sample(c, sensor, qmin, qmax, kind)
    if full is None:
        sc = (qmin, qmax)
    else:
        sc = (float(np.percentile(full, 2)), float(np.percentile(full, 98)))
    _psd_scale_cache[(stat, sensor)] = sc
    return sc


def _psd_sample(c, sensor, qmin, qmax, kind=None):
    """A few hundred spectra spread across the sensor's range, as dBm values
    (offset applied) -- the sample behind both the colour scale and the
    summary-matching LUT below. None when the sensor has nothing."""
    if (kind or _psd_kind) == "chunk":
        # Four chunks spread across the sensor, picked by timestamp.
        # USING SAMPLE applies to the table before WHERE does, so sampling 4
        # rows out of every chunk in the file and *then* keeping this sensor's
        # returns nothing at all once there is more than a handful of sensors.
        # The empty result fell through to (qmin, qmax) below -- the full 90 dB
        # quantization range against a signal spanning about 50 -- so every
        # tile came out nearly one flat colour and the layer looked frozen.
        span = c.execute("SELECT min(t0), max(t0) FROM psd_chunk WHERE sensor=?",
                         [sensor]).fetchone()
        rows = []
        if span and span[0] is not None:
            lo, hi = span
            for k in range(4):
                r = c.execute("SELECT n, times, specs FROM psd_chunk "
                              "WHERE sensor=? AND t0>=? ORDER BY t0 LIMIT 1",
                              [sensor, lo + (hi - lo) * k / 4.0]).fetchone()
                if r:
                    rows.append(r)
        specs = _unpack(rows, NF)[1] if rows else None
    else:
        # A reservoir sample over the row schema reads every row in the table:
        # 12 s on a 400 MB test database, minutes on the multi-GB real one, and
        # it is paid by the FIRST zoom into this layer -- which reads as the
        # layer being broken. Short reads spread across the sensor's time range
        # cost milliseconds and give the same 2nd/98th percentiles.
        rng = c.execute("SELECT t_min, t_max FROM psd_meta WHERE sensor=?",
                        [sensor]).fetchone()
        blobs = []
        if rng and rng[0] is not None and rng[1] is not None:
            step = max((rng[1] - rng[0]) / SCALE_PROBES, 1.0)
            for k in range(SCALE_PROBES):
                a = rng[0] + k * step
                blobs += [r[0] for r in c.execute(
                    "SELECT spec FROM psd WHERE sensor=? AND t>=? AND t<? "
                    "ORDER BY t LIMIT ?", [sensor, a, a + step, SCALE_PER_PROBE]
                ).fetchall()]
        specs = (np.frombuffer(b"".join(blobs), np.uint8).reshape(len(blobs), NF)
                 if blobs else None)
    if specs is None or not len(specs):
        return None
    specs = specs[:: max(1, len(specs) // 400)]
    return qmin + (specs.astype(np.float32) / 255.0) * (qmax - qmin) + PSD_DBM_OFFSET


# Summary->PSD colour matching. The two layers show the same signal on the
# same dBm axis, but their value distributions differ (channel power vs
# max-pooled PSD), so identical colour ramps still look shifted at the layer
# boundary. Quantile matching -- map each PSD value to the summary value of
# equal rank, per sensor, from the data itself -- pins the whole distribution,
# not just two endpoints. Deterministic, cached, ~200 summary rows + the
# existing PSD probe per sensor. Set ATLAS_NO_COLORMATCH=1 to disable.
_match_cache = {}


# Which summary column each display statistic matches onto. The match used to
# target mx no matter what the SHOW buttons said, so a user viewing the median
# summary and zooming into PSD got tiles matched onto the max distribution --
# measured as a built-in ~7 dB legend jump at the handoff. The PSD data itself
# is always the max detector; matching it onto the DISPLAYED stat's quantiles is
# what makes the colours continuous across the switch, which is the whole point
# of the match.
_STAT_COL = {"max": "mx", "median": "md", "mean": "mn"}


def _psd_match(c, sensor, qmin, qmax, stat="max", kind=None):
    if connection is None or os.environ.get("ATLAS_NO_COLORMATCH"):
        return None
    col = _STAT_COL.get(stat, "mx")
    m = _match_cache.get((sensor, col))
    if m is not None:
        return m or None                    # () = known no-match, cached
    try:
        # /DBM_DIV for the same reason /api/heatmap does it: this is the scale the
        # PSD layer is being matched ONTO, so it has to be the dBm the summary
        # layer actually draws. Reading it raw off a compacted database made every
        # matched PSD value, and the legend derived from it, 10x too large.
        s = np.array([r[0] for r in connection.cursor().execute(
            f"SELECT {col} FROM lvl_h1 WHERE sensor=? AND {col} IS NOT NULL",
            [sensor]).fetchall()], np.float32) / DBM_DIV
    except Exception as e:
        print(f"[serve] colour match off for {sensor}: {e}")
        s = np.empty(0)
    p = _psd_sample(c, sensor, qmin, qmax, kind)
    if len(s) < 100 or p is None or p.size < 100:
        _match_cache[(sensor, col)] = ()
        return None
    q = np.linspace(0.0, 100.0, 65)
    pq = np.percentile(p, q).astype(np.float64)
    sq = np.percentile(s, q).astype(np.float64)
    pq += np.arange(65) * 1e-4              # strictly increasing for interp
    m = (pq, sq, float(np.percentile(s, 2)), float(np.percentile(s, 98)))
    _match_cache[(sensor, col)] = m
    return m


def _apply_match(img, match, qmin, qmax, offset):
    """Remap img (dBm) onto the summary's dBm ramp -- same numbers as
    np.interp(img, match[0], match[1]), by a faster path.

    img comes from a uint8-quantized, max-pooled grid, so however large the
    tile is it only ever holds the 256 values that quantization allows (plus
    NaN in gap columns). np.interp does a binary search per pixel to place
    each of those into the 65-point match curve; building that mapping once
    for the 256 possible inputs and gathering by index gets the identical
    number without repeating the search millions of times. Verified
    byte-for-byte equal to np.interp on real PSD tiles, gaps included.
    """
    v = np.arange(256, dtype=np.float64)
    lut = np.interp(qmin + (v / 255.0) * (qmax - qmin) + offset,
                    match[0], match[1]).astype(np.float32)
    finite = np.isfinite(img)
    idx = np.zeros(img.shape, dtype=np.uint8)
    if finite.any():
        scaled = (img[finite] - offset - qmin) * (255.0 / (qmax - qmin))
        idx[finite] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    out = lut[idx]
    out[~finite] = np.nan
    return out


def _pfp_nearest(c, sensor, freq, t1, npos):
    """The one frame to hold when the window itself is empty.

    The PSD layer has had this since the deep-zoom work; the frame layer needs
    it more, not less. A sweep dwells on each channel once every ~90 s, so any
    window shorter than that -- which is most of the frame layer's useful range
    -- can easily contain no capture for the channel on screen at all. Without a
    frame to hold, that window renders as nothing and zooming in far enough
    turns the layer blank rather than steady.
    """
    if _pfp_kind == "chunk":
        row = c.execute("SELECT n, times, frames FROM pfp_chunk WHERE sensor=? "
                        "AND freq=? AND t0<? ORDER BY t0 DESC LIMIT 1",
                        [sensor, freq, t1]).fetchone()
        if row is None:
            row = c.execute("SELECT n, times, frames FROM pfp_chunk WHERE "
                            "sensor=? AND freq=? ORDER BY t0 LIMIT 1",
                            [sensor, freq]).fetchone()
        if row is None:
            return None
        ts, frames = _unpack([row], npos)
        i = max(0, int(np.searchsorted(ts, t1)) - 1)
        return frames[i:i + 1]
    row = c.execute("SELECT t, frame FROM pfp WHERE sensor=? AND freq=? AND t<? "
                    "ORDER BY t DESC LIMIT 1", [sensor, freq, t1]).fetchone()
    if row is None:
        row = c.execute("SELECT t, frame FROM pfp WHERE sensor=? AND freq=? "
                        "ORDER BY t LIMIT 1", [sensor, freq]).fetchone()
    if row is None:
        return None
    return _unpack_rows([row], npos)[1]


def _psd_count(c, sensor, t0, t1, kind=None):
    """Captures stored in [t0,t1) -- only used to decide whether to stride, so
    the chunk estimate (whole chunks that overlap) is close enough."""
    if (kind or _psd_kind) == "chunk":
        r = c.execute("SELECT coalesce(sum(n),0) FROM psd_chunk WHERE sensor=? "
                      "AND t1>=? AND t0<?", [sensor, t0, t1]).fetchone()
    else:
        r = c.execute("SELECT count(*) FROM psd WHERE sensor=? AND t>=? AND t<?",
                      [sensor, t0, t1]).fetchone()
    return int(r[0] or 0)


def _psd_nearest(c, sensor, t1, kind=None):
    """The one capture to hold when the window itself is empty."""
    if (kind or _psd_kind) == "chunk":
        row = c.execute("SELECT n, times, specs FROM psd_chunk WHERE sensor=? AND t0<? "
                        "ORDER BY t0 DESC LIMIT 1", [sensor, t1]).fetchone()
        if row is None:
            row = c.execute("SELECT n, times, specs FROM psd_chunk WHERE sensor=? "
                            "ORDER BY t0 LIMIT 1", [sensor]).fetchone()
        if row is None:
            return None
        ts, specs = _unpack([row], NF)
        i = max(0, int(np.searchsorted(ts, t1)) - 1)
        return specs[i:i + 1]
    row = c.execute("SELECT t, spec FROM psd WHERE sensor=? AND t<? "
                    "ORDER BY t DESC LIMIT 1", [sensor, t1]).fetchone()
    if row is None:
        row = c.execute("SELECT t, spec FROM psd WHERE sensor=? ORDER BY t LIMIT 1",
                        [sensor]).fetchone()
    if row is None:
        return None
    return _unpack_rows([row], NF)[1]


# Full-band window grids, keyed by (stat, sensor, t0, t1, cols). A frequency zoom
# keeps the time window fixed, so every band slice of the same window can be
# cut from one already-pooled grid instead of re-reading and re-inflating
# every chunk (the whole cost of a wide tile). One grid is cols x NF uint8
# (~7 MB); four of them bound the cache well under 30 MB.
#
# Four was one statistic's budget. The key gained `stat` when median and mean
# arrived, so the three databases -- same sensors, same windows -- now share
# these slots: toggling statistic at two windows exhausted the cache and every
# toggle re-read and re-pooled a window that had already been pooled. Slots are
# allotted per statistic ON DISK instead, which keeps the behaviour above true
# for whichever one is on screen and leaves a max-only install at exactly the
# four it always had.
#
# The honest ceiling, since the "well under 30 MB" above describes one
# statistic at typical widths: an entry is cols x NF bytes, so 2.7 MB at a
# 1200-column tile and 9.2 MB at MAX_TILE_COLS (4096). Three statistics at
# three slots each is 9 entries -- ~25 MB typical, ~83 MB if every one of them
# is a full-width export. Three rather than four per statistic keeps that
# worst case near where it was.
_PWIN_PER_STAT = 3
_PWIN_CACHE, _PWIN_MAX = {}, _PWIN_PER_STAT
_pwin_lock = threading.Lock()


def _pick_psd_level(span, cols, ncap, levels=None):
    """Which coarse bucket to draw this window from, or None for the captures.

    Two guards, both learned the hard way by measuring against a no-thinning
    render of the same window:

    * A bucket must be no wider than a pixel column, so a column is built from
      whole buckets rather than a fraction of one.
    * The capture path must be one that would THIN. Below about three weeks it
      reads every capture and is therefore exactly right, so a level there
      could only ever make the answer worse, however fast it is.

    Where both hold, the level is both faster and closer to the true max than
    reading one capture in N -- which is the only reason to prefer it.
    """
    if levels is None:
        levels = _psd_lvls
    if not levels or cols <= 0 or span <= 0:
        return None
    if _stride_for(ncap, cols) <= 1:
        return None                       # captures are exact here; leave them
    col = span / cols
    usable = [b for b in levels if b <= col]
    return max(usable) if usable else None


# A coarse index built in RAM, for databases that have never had
# build_psd_levels.py run on them. Same rows the on-disk table would hold -- one
# bin-wise max spectrum per bucket -- computed by one background pass over a
# sensor's captures the first time anything asks about its PSD layer, which the
# viewer does on page load. So the expensive wide window is usually ready before
# the user can zoom to it, and nothing on disk had to change.
#
# On-disk levels are still better and are preferred whenever present: they cost
# no RAM and survive a restart. This is the fallback that makes that build step
# optional rather than required.
MEM_BUCKET = 3600
# Entries held, keyed (stat, sensor); ~22 MB per sensor-year. Three was three
# SENSORS when max was the only statistic. Now one sensor viewed in all three
# statistics fills the whole budget on its own, so a second sensor evicted the
# first's finished index and any return to it rebuilt from scratch. Sized per
# statistic present on disk instead, which leaves a max-only install at 3.
#
# Two per statistic, not three: at 22 MB per sensor-year an entry is far and
# away the most expensive thing cached here, and this index only exists at all
# when the on-disk pyramid is missing -- build_psd_levels.py makes the whole
# subsystem moot. Six entries is ~130 MB per year of data on a three-statistic
# set, against ~66 MB before.
_MEM_PER_STAT = 2
_MEM_MAX = _MEM_PER_STAT
_lvl_short = set()        # (stat, sensor) whose on-disk pyramid did not cover a
                          # window we were asked for: build the in-RAM one anyway
_memlvl = {}              # (stat, sensor) -> dict | 'queued' | 'building'
                          #                 | ('failed', monotonic)
_mem_queue = []           # (stat, sensor) waiting, in arrival order
_mem_workers = 0          # live build threads; see _memlvl_worker
_mem_lock = threading.Lock()


def _scale_stat_budgets():
    """Re-size the stat-keyed caches to the statistics actually on disk.

    Called whenever a statistic's database is opened for the first time, which
    is the only moment the count can change. Never shrinks below the historical
    single-statistic budget.
    """
    global _PWIN_MAX, _MEM_MAX
    n = max(1, sum(1 for s in PSD_LAYER_STATS if os.path.exists(psd_db_path(s))))
    # Never below the budgets a max-only install has always had (4 and 3): the
    # per-statistic allotments are smaller than those, so plain multiplication
    # SHRANK the single-statistic case -- three sensors' worth of in-RAM index
    # became two, and returning to the first rebuilt it from scratch. Growing
    # the multi-statistic case must not cost the installs that never changed.
    _PWIN_MAX = max(_PWIN_PER_STAT * n, 4)
    _MEM_MAX = max(_MEM_PER_STAT * n, 3)


# The build is a background convenience, never a reason for a request the user IS
# waiting on to get slower. Measured on a 2-core box: an unthrottled build turned
# an 18 s first wide window into 56 s by competing for the same cores. After each
# batch the thread sleeps for a share of the time that batch took, which caps it
# near 60% of one core and leaves the tile path the rest. Costs the build about
# half again in wall clock, which nobody is watching.
MEM_THROTTLE = 0.7


def _memlvl_build(sensor, stat="max"):
    """One pass over one statistic's captures -> bucketed spectra, in RAM.

    The pooling inside a bucket is max for every statistic: for the median and
    mean layers the stored per-capture values ARE medians/means (computed at
    the sensor over 320k FFTs); pooling those to one value per bucket for a
    pixel keeps the strongest capture visible, the same display convention the
    capture path itself uses when several captures share a pixel column.
    """
    try:
        connection_s, kind_s, _lv = psd_stat_info(stat)
        c = connection_s.cursor()
        buckets = {}
        if kind_s == "chunk":
            cur = c.execute("SELECT n, times, specs FROM psd_chunk WHERE sensor=? "
                            "ORDER BY t0", [sensor])
            batch = 32
        else:
            cur = c.execute("SELECT t, spec FROM psd WHERE sensor=? ORDER BY t",
                            [sensor])
            batch = 4096
        while True:
            t_batch = time.monotonic()
            rows = cur.fetchmany(batch)
            if not rows:
                break
            if kind_s == "chunk":
                pairs = [_unpack([r], NF) for r in rows]
            else:
                pairs = [(np.array([r[0] for r in rows], np.float64),
                          np.frombuffer(b"".join(r[1] for r in rows),
                                        np.uint8).reshape(len(rows), NF))]
            for ts, mat in pairs:
                keys = (ts // MEM_BUCKET).astype(np.int64)
                for k in np.unique(keys):
                    m = mat[keys == k].max(axis=0)
                    prev = buckets.get(k)
                    buckets[k] = m if prev is None else np.maximum(prev, m)
            time.sleep(min(0.25, (time.monotonic() - t_batch) * MEM_THROTTLE))
        if not buckets:
            with _mem_lock:
                _memlvl[(stat, sensor)] = _failed()
            return
        ks = np.array(sorted(buckets), np.int64)
        built = {"t": ks * MEM_BUCKET,
                 "smax": np.stack([buckets[k] for k in ks]),
                 "t0": float(ks[0] * MEM_BUCKET),
                 "t1": float((ks[-1] + 1) * MEM_BUCKET)}
        with _mem_lock:
            _memlvl[(stat, sensor)] = built
            _memlvl_evict(keep=(stat, sensor))
        print(f"[serve] coarse PSD {stat} index for {sensor} ready in RAM "
              f"({len(ks)} x {MEM_BUCKET}s buckets). ingest/build_psd_levels.py "
              f"stores this on disk instead, so it survives a restart.")
    except Exception as e:
        print(f"[serve] could not build the in-RAM PSD {stat} index for {sensor}: {e}")
        with _mem_lock:
            _memlvl[(stat, sensor)] = _failed()
            _memlvl_evict()


# How many indexes may build at once. MEM_THROTTLE caps ONE build near 60% of a
# core, measured on a two-core box; three statistics warmed together ran three of
# those at once and spent the whole margin the throttle exists to protect. But
# forcing them strictly one at a time is worse where there are cores to spare:
# the third statistic's index then lands three build-times later, and measured on
# this fixture that turned a warm 477 ms median tile back into a cold 3.2 s one.
# So: one build per spare core, never more than there are statistics, never fewer
# than one. A two-core box serialises (the calibration the throttle assumes); an
# eight-core workstation builds all three at once and still leaves five cores to
# the tile path. ATLAS_INDEX_WORKERS overrides it.
def _index_workers():
    try:
        n = int(os.environ.get("ATLAS_INDEX_WORKERS", "0"))
    except ValueError:
        n = 0
    if n > 0:
        return n
    return max(1, min(len(PSD_LAYER_STATS), (os.cpu_count() or 2) - 1))


# A failed build used to be a permanent verdict. One transient cause -- the
# database write-locked for a moment by an ingest, a momentary allocation failure
# -- disabled the in-RAM index for that (stat, sensor) for the life of the
# process, and with a single sensor and statistic the marker was never even
# evicted (len 1 is under the cap). Correctness was never affected, but every
# wide window silently went back to the slow path with nothing to say why. So the
# marker expires and the work is retried.
MEM_FAIL_COOLDOWN = 120.0


def _failed():
    return ("failed", time.monotonic())


def _is_failed(v):
    return isinstance(v, tuple) and v and v[0] == "failed"


def _failed_expired(v):
    return _is_failed(v) and (time.monotonic() - v[1]) > MEM_FAIL_COOLDOWN


def _memlvl_evict(keep=None):
    """Trim _memlvl to _MEM_MAX. Call with _mem_lock held.

    Never evicts the entry a build is working on: memlvl_warm's only guard is
    membership, so dropping it mid-build lets the next warm queue the same work
    again. A "queued" entry IS evictable -- nothing has been spent on it, and
    dropping it just means the capture path answers that window instead.

    Runs on the failure paths too, not only on success. Sitting on the success
    path alone, a run of failures (a database briefly write-locked by an ingest,
    say) filled the dict past its cap with 'failed' markers that nothing would
    ever clear.
    """
    for k in [k for k, v in _memlvl.items()
              if k != keep and _failed_expired(v)]:
        _memlvl.pop(k, None)                  # cooled off: retriable again
    while len(_memlvl) > _MEM_MAX:
        victim = next((k for k, v in _memlvl.items()
                       if k != keep and v != "building"), None)
        if victim is None:
            return
        _memlvl.pop(victim)
        if victim in _mem_queue:          # dropped before it ever started
            _mem_queue.remove(victim)


def _memlvl_worker():
    """Drain the queue, one build at a time per worker, then retire.

    Both `finally` blocks are load-bearing, and a harness that kills a build
    thread outright proved it: the count is what memlvl_warm tops the pool up
    against, so leaking it by even one leaves the pool permanently short -- and
    a thread that dies inside a build leaks it. Same for the entry it was
    building: left saying "building" it is never evicted and never retried, so
    that (stat, sensor) has no in-RAM index for the life of the process.
    """
    global _mem_workers
    try:
        while True:
            with _mem_lock:
                while _mem_queue:
                    key = _mem_queue.pop(0)
                    if _memlvl.get(key) == "queued":
                        break
                else:
                    return
                _memlvl[key] = "building"
            stat, sensor = key
            try:
                _memlvl_build(sensor, stat)
            except Exception as e:          # noqa: BLE001
                # One bad build must not end the worker. Letting it propagate
                # killed the thread and stranded everything still queued behind
                # it until some unrelated later request happened to top the pool
                # up -- measured with a harness that fails one build in five.
                print(f"[serve] PSD index build for {sensor}/{stat} failed: {e}")
            finally:
                with _mem_lock:
                    if _memlvl.get(key) == "building":
                        _memlvl[key] = _failed()    # died without a verdict
                    _memlvl_evict()
    finally:
        # Under the lock, so a warm racing this either sees the lower count and
        # starts a replacement, or enqueues before the check and is picked up by
        # the top-up. No window where a queued build is orphaned.
        with _mem_lock:
            _mem_workers -= 1


def memlvl_warm(sensor, stat="max", urgent=False):
    """Queue the background build once per (stat, sensor). Cheap to call often.

    `urgent` means a tile for this statistic is being drawn right now, so its
    index is what matters soonest: it goes to the front of the queue. Without
    this, a statistic the user has actually switched to waited behind whatever
    page load happened to queue first.
    """
    global _mem_workers
    connection_s, kind_s, levels_s = psd_stat_info(stat)
    if not sensor or connection_s is None or kind_s is None:
        return
    key = (stat, sensor)
    if levels_s and key not in _lvl_short:
        return                        # on-disk levels present and sufficient
    start = 0
    with _mem_lock:
        if _failed_expired(_memlvl.get(key)):
            _memlvl.pop(key, None)            # the cooldown has passed; retry it
        if key in _memlvl:
            if urgent and _memlvl.get(key) == "queued" and _mem_queue and _mem_queue[0] != key:
                _mem_queue.remove(key)          # already waiting: jump the queue
                _mem_queue.insert(0, key)
        elif len(_mem_queue) < _MEM_MAX:
            _memlvl[key] = "queued"
            _mem_queue.insert(0, key) if urgent else _mem_queue.append(key)
        # else: more waiting than we would keep. The capture path answers
        # correctly, just slower.
        #
        # Top the pool up even when nothing was queued THIS call. Checking the
        # worker count only after a successful enqueue meant workers that died
        # with a full queue could never be replaced: every later call bailed at
        # the capacity check, so the queue never drained and never fell below the
        # cap. A build thread can die that way on a BaseException, or on
        # Thread.start() failing under thread exhaustion.
        if _mem_queue:
            want = min(_index_workers(), len(_mem_queue) + 1)
            start = max(0, want - _mem_workers)
            _mem_workers += start
    # Started outside the lock so a start() failure cannot hold it, and so a slow
    # thread launch never blocks a request. A failed start is undone here and
    # recovered by the top-up on the next call.
    for _ in range(start):
        try:
            threading.Thread(target=_memlvl_worker, daemon=True,
                             name="psdlvl").start()
        except BaseException as e:     # out of threads or RAM: degrade, never 500
            # BaseException, not RuntimeError. Under memory pressure
            # Thread.start() raises MemoryError, which escaped -- the request
            # became a 500 and, worse, _mem_workers stayed incremented forever.
            # On a 2-core machine _index_workers() is 1, so start was then always
            # 0 and no worker was ever launched again: entries sat "queued" in
            # _memlvl for the life of the process and the whole in-RAM index
            # subsystem was dead. Measured: 0 builds over 46 requests.
            with _mem_lock:
                _mem_workers -= 1
            print(f"[serve] could not start a PSD index worker: {e!r}")


def memlvl_ready(sensor, t0, t1, stat="max"):
    """The in-RAM index for this statistic's sensor, if built and spanning."""
    with _mem_lock:
        m = _memlvl.get((stat, sensor))
    if not isinstance(m, dict):
        return None
    return m if (m["t0"] <= t0 and t1 <= m["t1"]) else None


_lvl_cov, _lvl_cov_lock = {}, threading.Lock()


def _lvl_covers(c, sensor, lvl, t0, t1, stat="max"):
    """Does this level actually span the whole window being asked for?

    build_psd_levels.py is a separate step, so the levels can legitimately stop
    short of the captures: the obvious way is ingesting a new month and not
    re-running it. Reading a level anyway would leave the newest part of the
    window with no rows at all, and the grid's sample-and-hold would paper over
    it with the last value it did have -- stale data presented as current, which
    is worse than the slow path by a wide margin. So check, and fall back.

    Cached per (statistic, sensor, bucket) for the life of the process. Coverage
    only grows -- build_psd_levels.py appends to psd_lvl and never removes rows --
    so a cached "covers it" cannot go stale in the unsafe direction. A cached
    "does not cover it" can, if the levels are rebuilt under a running server;
    restarting picks it up.
    """
    key = (stat, sensor, lvl)
    with _lvl_cov_lock:
        cov = _lvl_cov.get(key)
    if cov is None:
        try:
            r = c.execute("SELECT min(t), max(t) FROM psd_lvl WHERE sensor=? "
                          "AND bucket=?", [sensor, lvl]).fetchone()
        except Exception:
            r = None
        cov = (None, None) if not r or r[0] is None else (r[0], r[1] + lvl)
        with _lvl_cov_lock:
            _lvl_cov[key] = cov
    if cov[0] is None:
        return False
    return cov[0] <= t0 and t1 <= cov[1]


def _fill_from_levels(grid, rows, t0, t1, lvl, cols, ncap):
    """Pool coarse level rows onto the grid's columns.

    Every column takes the max of every bucket that OVERLAPS it, not just the
    one whose centre happens to land in it. Placing by centre looked right and
    measured badly -- 0.9 dB mean and 31 dB worst case against a no-thinning
    render -- because the rest of a straddling bucket's time was credited to a
    neighbouring column, so structure slid sideways by up to half a column.
    Overlap assignment can only ever over-include (a column's edges may carry
    part of the adjacent bucket, the same way the summary pyramid's cells do)
    and never moves or drops anything.
    """
    # One join + one frombuffer, not one frombuffer per row: at the finest level
    # a wide window is tens of thousands of 2250-byte rows, and building that
    # many little arrays and stacking them cost more than the query.
    ts = np.fromiter((r[0] for r in rows), np.float64, len(rows))
    # DuckDB hands these over as BLOBs; the in-RAM index already has arrays.
    if isinstance(rows[0][1], (bytes, bytearray, memoryview)):
        mat = np.frombuffer(b"".join(r[1] for r in rows),
                            np.uint8).reshape(len(rows), NF)
    else:
        mat = np.stack([r[1] for r in rows])
    span = max(t1 - t0, 1e-9)
    col_w = span / cols

    def put(ci, m):
        ok = (ci >= 0) & (ci < cols)
        if not ok.any():
            return
        ci, m = ci[ok], m[ok]
        # rows arrive ORDER BY t, so the column index is already non-decreasing
        # and reduceat can group in place; only sort if that ever stops holding.
        if ci.size > 1 and np.any(np.diff(ci) < 0):
            order = np.argsort(ci, kind="stable")
            ci, m = ci[order], m[order]
        starts = np.flatnonzero(np.r_[True, ci[1:] != ci[:-1]])
        pooled = np.maximum.reduceat(m, starts, axis=0)
        uniq = ci[starts]
        grid.buf[uniq] = np.maximum(grid.buf[uniq], pooled)
        grid.filled[uniq] = True

    first = np.floor((ts - t0) / col_w).astype(np.int64)
    last = np.floor((ts + lvl - 1e-9 - t0) / col_w).astype(np.int64)
    for k in range(int(math.ceil(lvl / col_w)) + 1):
        step = first + k
        sel = step <= last
        if not sel.any():
            break
        put(step[sel], mat[sel])
    # The readout says how many captures are behind the picture; that is still
    # the capture count, not the number of level rows it was summarised into.
    grid.ncap = ncap


def _psd_window(c, sensor, t0, t1, cols, fi0, fi1,
                stat="max", kind=None, levels=None):
    """Spectra in [t0,t1), pooled onto `cols` uniform time bins.

    Pools the FULL band once per (sensor, stat, window) and serves any
    requested band as a slice of that grid: the first tile of a window pays
    the scan, every frequency zoom inside it answers from memory.

    `stat` names which statistic's database the cursor belongs to; it is part
    of every cache key, because a max grid answering a median request would be
    silently wrong in the way nothing downstream could detect.

    Data gap -> hold the nearest capture (matches the summary layer's gap-fill,
    so zooming into a quiet stretch keeps showing last-known data).
    """
    if kind is None:
        kind = _psd_kind
    if levels is None:
        levels = _psd_lvls
    key = (stat, sensor, t0, t1, cols)
    with _pwin_lock:
        hit = _PWIN_CACHE.pop(key, None)
        if hit is not None:
            _PWIN_CACHE[key] = hit                      # LRU touch
    if hit is not None:
        return _pwin_slice(hit[0], fi0, fi1), hit[1]
    grid = _Grid(t0, t1, cols, NF)

    # A coarse level, when one is no wider than a pixel column. Reading a few
    # thousand 2250-byte rows instead of inflating every chunk in range is what
    # takes a full-span window from ~18 s to well under one, and it is also the
    # more faithful answer: the level row is the max over EVERY capture in its
    # bucket, where the capture path below keeps only every Nth one at these
    # widths. Sound because the pooling is a max for every statistic -- which
    # also means a zoomed-out "median" view is a max OF medians, not a median.
    # build_psd_levels.py's docstring has the measured size of that gap; it is
    # the one place that number lives.
    ncap = _psd_count(c, sensor, t0, t1, kind)   # metadata only, no BLOBs read
    lvl = _pick_psd_level(t1 - t0, cols, ncap, levels)
    if lvl is not None and not _lvl_covers(c, sensor, lvl, t0, t1, stat):
        lvl = None                             # levels are behind the captures
        # Remember it. memlvl_warm's only test was "does psd_lvl exist", so a
        # PARTIAL pyramid -- one interrupted build, or an ingest since the last
        # build_psd_levels run -- suppressed the in-RAM index while the on-disk
        # one was refused for the window at hand. What answered instead was the
        # strided capture path: measured 1.70 dB mean and 13.1 dB worst error with
        # 89% of pixels under-read, against 0.22 dB and 0% when psd_lvl was
        # absent entirely. A partial pyramid was worse than none.
        with _mem_lock:
            _lvl_short.add((stat, sensor))
    rows = None
    if lvl is not None:
        rows = c.execute("SELECT t, smax FROM psd_lvl WHERE sensor=? AND bucket=? "
                         "AND t>=? AND t<? ORDER BY t",
                         [sensor, lvl, t0 - lvl, t1]).fetchall()
    elif _stride_for(ncap, cols) > 1 and MEM_BUCKET <= (t1 - t0) / max(cols, 1):
        # No usable table on disk, but the in-RAM index may be built and is the
        # same thing. Guards are identical: only where the captures would thin,
        # and only when a bucket fits inside a pixel column.
        m = memlvl_ready(sensor, t0, t1, stat)
        if m is not None:
            sel = (m["t"] >= t0 - MEM_BUCKET) & (m["t"] < t1)
            if sel.any():
                lvl = MEM_BUCKET
                rows = list(zip(m["t"][sel].tolist(), m["smax"][sel]))
    if lvl is not None and rows:
        _fill_from_levels(grid, rows, t0, t1, lvl, cols, ncap)
        gap = not grid.filled.all()
        with _pwin_lock:
            _PWIN_CACHE[key] = (grid, gap)
            while len(_PWIN_CACHE) > _PWIN_MAX:
                _PWIN_CACHE.pop(next(iter(_PWIN_CACHE)))
        return _pwin_slice(grid, fi0, fi1), gap

    stride = _stride_for(ncap, cols)

    if kind == "chunk":
        cur = c.execute("SELECT t0, n, times, specs FROM psd_chunk WHERE sensor=? "
                        "AND t1>=? AND t0<? ORDER BY t0", [sensor, t0, t1])
        while True:
            rows = cur.fetchmany(READ_BATCH)
            if not rows:
                break
            for ts, mat in _INFLATE.map(
                    # stat IS part of the key: the three statistics' databases
                    # hold the same sensors with the same chunk timestamps, so
                    # without it whichever statistic was drawn first poisoned
                    # every other one for overlapping chunks -- reproduced on
                    # compacted databases, tiles byte-identical across stats.
                    lambda r: _inflate_cached(("psd", stat, sensor, r[0]),
                                              r[2], r[3], r[1], NF), rows):
                keep = (ts >= t0) & (ts < t1)
                if stride > 1:                      # thin inside the chunk too
                    every = np.zeros(len(ts), bool)
                    every[::stride] = True
                    keep &= every
                if keep.any():
                    grid.add(ts[keep], mat[keep])
    else:
        # Stride on rowid, not row_number(): a window function has to
        # materialise every row of the partition -- BLOBs included -- before it
        # can number them, which measured 3.8x slower and used over twice the
        # memory here, and would be several GB on the real database. rowid is a
        # plain streaming filter. Rows go in ordered by time, so every Nth rowid
        # is also an even spread in time.
        sql = "SELECT t, spec FROM psd WHERE sensor=? AND t>=? AND t<? ORDER BY t"
        args = [sensor, t0, t1]
        if stride > 1:
            sql = ("SELECT t, spec FROM psd WHERE sensor=? AND t>=? AND t<? "
                   "AND rowid % ? = 0 ORDER BY t")
            args.append(stride)
        try:
            cur = c.execute(sql, args)
        except duckdb.Error:            # no rowid on this build: correct, slower
            cur = c.execute("SELECT t, spec FROM psd WHERE sensor=? AND t>=? "
                            "AND t<? ORDER BY t", [sensor, t0, t1])
        while True:
            rows = cur.fetchmany(READ_BATCH * PSD_ROWS_PER_BATCH)
            if not rows:
                break
            ts = np.array([r[0] for r in rows], np.float64)
            mat = np.frombuffer(b"".join(r[1] for r in rows),
                                np.uint8).reshape(len(rows), NF)
            grid.add(ts, mat)

    gap = False
    if not grid.filled.any():
        specs = _psd_nearest(c, sensor, t1, kind)
        if specs is None:
            return None, False
        grid = _Grid(t0, t1, 1, NF)
        grid.add(np.array([t0], np.float64), specs)
        gap = True
    with _pwin_lock:
        _PWIN_CACHE[key] = (grid, gap)
        while len(_PWIN_CACHE) > _PWIN_MAX:
            _PWIN_CACHE.pop(next(iter(_PWIN_CACHE)))
    return _pwin_slice(grid, fi0, fi1), gap


def _pwin_slice(full, fi0, fi1):
    """A band view of a cached full-band grid (shares the buffer, no copy)."""
    if fi0 == 0 and fi1 == NF - 1:
        return full
    sub = _Grid(full.t0, full.t0 + full.span, full.cols, 1)
    sub.buf = full.buf[:, fi0:fi1 + 1]
    sub.filled, sub.ncap = full.filled, full.ncap
    return sub


@app.route("/api/psd_meta")
def psd_meta():
    stat = request.args.get("stat", "max")
    if stat not in PSD_LAYER_STATS:
        stat = "max"                       # same rule as /api/psd_layer
    database, kind, _levels = psd_stat_info(stat)
    if database is None or kind is None:
        return jsonify({"has": False})
    sensor = request.args.get("sensor")
    # The viewer calls this on page load and on every sensor change, which is the
    # earliest moment we know which sensor matters -- so the coarse index starts
    # building while the user is still looking at the summary layer.
    #
    # EVERY statistic this sensor has, not just the requested one. The viewer
    # asks without a stat, which resolved to max, so a median or mean database
    # missing its on-disk levels started its index only when the user first
    # clicked that button -- the one moment it competes with the tile they are
    # waiting for. memlvl_warm returns immediately when the on-disk pyramid is
    # there, so on a properly built set this loop costs nothing.
    stats_here = psd_stats_available(sensor)
    row = database.cursor().execute("SELECT f0, df, nf, t_min, t_max FROM psd_meta WHERE sensor=?",
                             [sensor]).fetchone()
    # Deliberately NOT falling back to a sibling statistic's time range when
    # max's is null. Tried it; examples/test_ingest.py caught why it is wrong: a
    # max database that is an empty stub while median holds real data would
    # advertise the layer, the viewer would enable it with max still selected,
    # and every tile would come back empty -- exactly the silent "layer never
    # draws" failure the null-range check below exists to prevent.
    # An ingest that read no files leaves a meta row with a null time range.
    # Never advertise a layer whose backing rows are not actually there.
    if not row or row[3] is None or row[4] is None:
        return jsonify({"has": False})
    f0, df, nf, tmin, tmax = row
    # Only once the layer is known to be real: warming a statistic for a sensor
    # whose layer is about to be reported absent is background work for a picture
    # nobody will ask for.
    for s in stats_here:
        memlvl_warm(sensor, s)
    # `stats`: every statistic that has a database holding THIS sensor, so the
    # viewer can light up exactly the SHOW buttons that will actually answer.
    return jsonify({"has": True, "fmin": f0, "fmax": f0 + (nf - 1) * df,
                    "t_min": tmin, "t_max": tmax,
                    "stats": stats_here})


@app.route("/api/psd_layer")
def psd_layer():
    sensor = request.args.get("sensor")
    t0, t1, H = _window()
    f0 = _num("f0", 0.0); f1 = _num("f1", 0.0)
    fi0 = int(round((f0 - F0) / DF)) if f0 > 0 else 0
    fi1 = int(round((f1 - F0) / DF)) if f1 > 0 else NF - 1
    fi0 = max(0, min(NF - 1, fi0)); fi1 = max(0, min(NF - 1, fi1))
    if fi1 <= fi0:
        fi0, fi1 = 0, NF - 1
    # Unknown statistics coerce to max rather than erroring -- a viewer must
    # never lose its picture to a typo -- but the substitution is DECLARED: the
    # tile's X-Meta carries the statistic actually rendered, and /api/psd_meta
    # applies the same rule, so the two endpoints agree about invalid input.
    stat = request.args.get("stat", "max")
    if stat not in PSD_LAYER_STATS:
        stat = "max"
    database, kind, levels = psd_stat_info(stat)
    if database is None or kind is None:
        return jsonify({"error": f"psd {stat} layer not ready"}), 503
    # urgent: this statistic is being drawn now, so its index outranks anything
    # a page load queued speculatively.
    memlvl_warm(sensor, stat, urgent=True)
    W = max(1, _num("w", 1200, int))
    cursor = database.cursor()
    grid, gap = _psd_window(cursor, sensor, t0, t1, _grid_cols(W), fi0, fi1,
                            stat=stat, kind=kind, levels=levels)
    mrow = cursor.execute("SELECT qmin, qmax FROM psd_meta WHERE sensor=?", [sensor]).fetchone()
    if grid is None:
        return jsonify({"error": "no psd data for this sensor"}), 404
    qmin, qmax = mrow or (-180.0, -90.0)
    img = grid.image(qmin, qmax, H, PSD_DBM_OFFSET)   # (nF, cols); row 0 = low f
    match = _psd_match(cursor, sensor, qmin, qmax, stat, kind)
    if match is not None:                 # remap onto the summary's dBm ramp
        img = _apply_match(img, match, qmin, qmax, PSD_DBM_OFFSET)
    locked = _locked_scale()
    if locked:
        vmin, vmax = locked
    elif match is not None:               # scale in summary terms too
        vmin, vmax = match[2], match[3]
    else:
        vmin, vmax = _psd_scale(cursor, sensor, qmin, qmax, stat, kind)  # full-range: no drift
    return _tile(_colorize(img, vmin, vmax, request.args.get("cmap", "inferno")), {
        "t0": t0, "t1": t1, "stat": stat,
        "fmin": F0 + fi0 * DF, "fmax": F0 + fi1 * DF,
        "ncap": int(grid.ncap), "cols": int(img.shape[1]), "nf": int(img.shape[0]),
        "vmin": round(vmin, 1), "vmax": round(vmax, 1), "gap": gap,
    })


# ---- PFP layer (periodic-frame-power, ~18 us within a 10 ms frame) ----
PFP_DB = os.environ.get("PFP_DB") or os.path.join(DB_DIR, "pfp.duckdb")
_pfp_connection = None
_pfp_freqs_cache = {}
PFP_SNAP_HZ = 10e6      # how far a requested channel may be off and still resolve
_pfp_kind = None        # 'chunk' | 'rows' | None
_pfp_init = threading.Lock()


def pfp_connection():
    global _pfp_connection, _pfp_kind
    if _pfp_connection is not None or not os.path.exists(PFP_DB):
        return _pfp_connection
    with _pfp_init:                 # same first-load race as PSD above
        if _pfp_connection is not None:
            return _pfp_connection
        try:
            connection = duckdb.connect(PFP_DB, read_only=True)
        except Exception as e:
            # Silence here meant the layer simply was not there: /api/pfp_meta
            # answers {"has": false}, the viewer switches the layer off, and
            # nothing anywhere says why. The usual causes are an ingest or
            # compact_db still holding the write lock, and a file that is not a
            # DuckDB database -- both worth naming.
            print(f"[serve] cannot open {os.path.basename(PFP_DB)}: {e}")
            print("[serve]   -- the PFP layer will not be available.")
            return None
        t = table_names(connection)            # same dual-schema rule as PSD above
        _pfp_kind = "chunk" if "pfp_chunk" in t else ("rows" if "pfp" in t else None)
        if _pfp_kind == "rows":
            print("[serve] pfp.duckdb is the uncompacted row schema; serving it "
                  "directly. Run ingest/compact_db.py to shrink it.")
        _pfp_connection = connection                  # publish last (kind first; see psd_connection)
    return _pfp_connection


@app.route("/api/pfp_meta")
def pfp_meta():
    sensor = request.args.get("sensor")
    database = pfp_connection()
    if database is None or _pfp_kind is None:
        return jsonify({"has": False})
    cursor = database.cursor()
    m = cursor.execute("SELECT npos, frame_ms, t_min, t_max, stat FROM pfp_meta WHERE sensor=?",
                    [sensor]).fetchone()
    if not m or m[2] is None or m[3] is None:
        return jsonify({"has": False})
    freqs = _pfp_freqs(cursor, sensor)
    return jsonify({"has": True, "npos": m[0], "frame_ms": m[1],
                    "t_min": m[2], "t_max": m[3], "stat": m[4], "freqs": freqs})


def _pfp_freqs(c, sensor):
    """The channel centres this sensor actually has PFP for (cached)."""
    freqs = _pfp_freqs_cache.get(sensor)
    if freqs is None:
        src = "pfp_chunk" if _pfp_kind == "chunk" else "pfp"
        freqs = [r[0] for r in c.execute(
            f"SELECT DISTINCT freq FROM {src} WHERE sensor=? ORDER BY 1",
            [sensor]).fetchall()]
        _pfp_freqs_cache[sensor] = freqs
    return freqs


def _pfp_snap(c, sensor, freq):
    """Snap a requested channel onto one this sensor really has.

    `WHERE freq = ?` is an exact match on a DOUBLE, so a caller whose channel
    grid is a hair off -- or off by a whole channel -- got an empty result and
    a 404, which the viewer could only read as "no frame layer here". Snapping
    to the nearest stored centre means the deepest layer still appears, and the
    tile reports which channel it actually is.

    Bounded to one channel spacing: a request that is merely on the wrong grid
    is worth rescuing, but answering one 3.5 GHz away with a real channel's
    data would be a confident lie about what is on screen.
    """
    freqs = _pfp_freqs(c, sensor)
    if not freqs:
        return None
    ch = min(freqs, key=lambda f: abs(f - freq))
    return ch if abs(ch - freq) <= PFP_SNAP_HZ else None


@app.route("/api/pfp_frame")
def pfp_frame():
    """Frame heatmap for one channel: X = capture time, Y = frame position."""
    sensor = request.args.get("sensor")
    freq = _num("freq", required=True)
    t0, t1, H = _window()
    W = max(1, _num("w", 1200, int))
    database = pfp_connection()
    if database is None or _pfp_kind is None:
        return jsonify({"error": "pfp not ready"}), 503
    cursor = database.cursor()
    m = cursor.execute("SELECT npos, frame_ms, qmin, qmax FROM pfp_meta WHERE sensor=?",
                  [sensor]).fetchone()
    if not m:
        return jsonify({"error": "no pfp for this sensor"}), 404
    npos, frame_ms, qmin, qmax = m
    ch = _pfp_snap(cursor, sensor, freq)
    if ch is None:
        return jsonify({"error": f"no pfp channel near {freq/1e6:.1f} MHz"}), 404
    grid = _Grid(t0, t1, _grid_cols(W), npos)
    if _pfp_kind == "chunk":
        cursor = cursor.execute(
            "SELECT t0, n, times, frames FROM pfp_chunk WHERE sensor=? AND freq=? "
            "AND t1>=? AND t0<? ORDER BY t0", [sensor, ch, t0, t1])
        while True:
            rows = cursor.fetchmany(READ_BATCH)
            if not rows:
                break
            for ts, mat in _INFLATE.map(
                    lambda r: _inflate_cached(("pfp", sensor, ch, r[0]),
                                              r[2], r[3], r[1], npos), rows):
                keep = (ts >= t0) & (ts < t1)
                if keep.any():
                    grid.add(ts[keep], mat[keep])
    else:
        cursor = cursor.execute(
            "SELECT t, frame FROM pfp WHERE sensor=? AND freq=? "
            "AND t>=? AND t<? ORDER BY t", [sensor, ch, t0, t1])
        while True:
            rows = cursor.fetchmany(READ_BATCH * PSD_ROWS_PER_BATCH)
            if not rows:
                break
            ts = np.array([r[0] for r in rows], np.float64)
            mat = np.frombuffer(b"".join(r[1] for r in rows),
                                np.uint8).reshape(len(rows), npos)
            grid.add(ts, mat)
    if not grid.filled.any():
        # Nothing for this channel inside the window: hold the frame that was
        # last measured before it, exactly as the PSD layer does, so zooming
        # past the ~90 s sweep interval goes steady rather than blank.
        held = _pfp_nearest(cursor, sensor, ch, t1, npos)
        if held is None:
            return jsonify({"error": "no pfp in window"}), 404
        grid.add(np.array([t0], np.float64), held)
    img = grid.image(qmin, qmax, H)        # (nrows, cols); row 0 = frame start
    locked = _locked_scale()
    if locked:
        vmin, vmax = locked
    else:
        # Every 8th finite value, not all 1.3M of them: two exact percentiles
        # cost 9 ms of partitioning per request and the 2nd/98th of a tile this
        # size are the same numbers either way (checked on HU: -88.59/-87.18
        # both ways). The gap columns stay excluded.
        seen = img[np.isfinite(img)][::8]
        vmin = float(np.percentile(seen, 2)); vmax = float(np.percentile(seen, 98))
    return _tile(_colorize(img, vmin, vmax, request.args.get("cmap", "inferno")), {
        "t0": t0, "t1": t1, "ncap": int(grid.ncap),
        "npos": npos, "nrows": int(img.shape[0]), "cols": int(img.shape[1]),
        "frame_ms": frame_ms, "freq": ch,
        "vmin": round(vmin, 1), "vmax": round(vmax, 1),
    })


# ---- IQ capture mode (independent SigMF/TDMS captures; own axes) ------
# Per-request read-only connections on purpose: iq_ingest.py may be appending
# new captures between requests, and a held handle would block its write lock.
IQ_DB = os.environ.get("IQ_DB") or os.path.join(DB_DIR, "iq.duckdb")


def iq_connection():
    if not os.path.exists(IQ_DB):
        return None
    try:
        return duckdb.connect(IQ_DB, read_only=True)
    except Exception:
        return None


@app.route("/api/iq_index")
def iq_index():
    database = iq_connection()
    if database is None:
        return jsonify({"captures": []})
    rows = database.execute("SELECT id, name, dataset, fc, fs, duration FROM iq_meta "
                     "ORDER BY dataset, name").fetchall()
    database.close()
    return jsonify({"captures": [
        {"id": r[0], "name": r[1], "dataset": r[2], "fc": r[3], "fs": r[4],
         "duration": r[5]} for r in rows]})


@app.route("/api/iq_meta")
def iq_meta():
    database = iq_connection()
    row = database and database.execute(
        "SELECT id, dataset, name, fc, fs, duration, n_samples, nfft, nfreq, "
        "hop, nlevels, vmin, vmax FROM iq_meta WHERE id=?",
        [request.args.get("id")]).fetchone()
    if database:
        database.close()
    if not row:
        return jsonify({"has": False})
    k = ["id", "dataset", "name", "fc", "fs", "duration", "n_samples", "nfft",
         "nfreq", "hop", "nlevels", "vmin", "vmax"]
    return jsonify({"has": True, **dict(zip(k, row))})


@app.route("/api/iq_layer")
def iq_layer():
    """WebP tile of one capture's spectrogram for a [t0,t1]x[f0,f1] window.
    Picks the finest pyramid level with <= ~2*w columns (mirrors pick_level)."""
    cid = request.args.get("id")
    database = iq_connection()
    if database is None:
        return jsonify({"error": "iq layer not ready"}), 503
    m = database.execute("SELECT fc, fs, duration, nfft, hop, nlevels, qmin, qmax, "
                  "vmin, vmax FROM iq_meta WHERE id=?", [cid]).fetchone()
    if not m:
        database.close()
        return jsonify({"error": "unknown capture"}), 404
    fc, fs, dur, nfft, hop, nlevels, qmin, qmax, dvmin, dvmax = m
    t0 = max(0.0, _num("t0", 0.0))
    t1 = min(dur, _num("t1", dur))
    W = max(1, _num("w", 1200, int))
    H = max(1, _num("h", 600, int))
    fmin_full, fmax_full = fc - fs / 2, fc + fs / 2
    f0 = max(fmin_full, _num("f0", fmin_full))
    f1 = min(fmax_full, _num("f1", fmax_full))
    if t1 <= t0 or f1 <= f0:
        database.close()
        return jsonify({"error": "empty window"}), 400
    span_cols0 = (t1 - t0) * fs / hop
    level = 0
    while level < nlevels - 1 and span_cols0 / (1 << level) > 2 * W:
        level += 1
    colw = hop * (1 << level) / fs                    # seconds per column
    c0 = int(t0 / colw)
    c1 = max(c0 + 1, int(math.ceil(t1 / colw)))
    rows = database.execute(
        "SELECT col0, ncols, chunk FROM iq_stft WHERE id=? AND level=? "
        "AND col0 < ? AND col0 + ncols > ? ORDER BY col0",
        [cid, level, c1, c0]).fetchall()
    database.close()
    if not rows:
        return jsonify({"error": "no data in window"}), 404
    mat = np.concatenate([np.frombuffer(r[2], dtype=np.uint8).reshape(r[1], nfft)
                          for r in rows])                       # (cols, nfft)
    base = rows[0][0]
    c0 = max(c0, base); c1 = min(c1, base + mat.shape[0])
    mat = mat[c0 - base:c1 - base]
    df = fs / nfft                                    # row 0 (fftshifted) = fc - fs/2
    fi0 = max(0, int((f0 - fmin_full) / df))
    fi1 = min(nfft, max(fi0 + 1, int(math.ceil((f1 - fmin_full) / df))))
    band = mat[:, fi0:fi1].astype(np.float32).T       # (nbins, ncols) row0 = low f
    nbins = band.shape[0]
    fb = max(1, math.ceil(nbins / H))
    pad = (-nbins) % fb
    if pad:
        band = np.pad(band, ((0, pad), (0, 0)))
    img = band.reshape(band.shape[0] // fb, fb, -1).max(axis=1)
    img = qmin + (img / 255.0) * (qmax - qmin)        # uint8 -> dBm
    # locked scale: no drift on zoom
    vmin = _num("vmin", dvmin)
    vmax = _num("vmax", dvmax)
    return _tile(_colorize(img, vmin, vmax, request.args.get("cmap", "inferno")), {
        "t0": c0 * colw, "t1": c1 * colw,
        "fmin": fmin_full + fi0 * df, "fmax": fmin_full + fi1 * df,
        "cols": int(img.shape[1]), "nf": int(img.shape[0]), "level": level,
        "vmin": round(vmin, 1), "vmax": round(vmax, 1),
    })


if __name__ == "__main__":
    port = int(os.environ.get("SEA_PORT", "8090"))
    try:
        from waitress import serve           # production WSGI: keep-alive, faster
        print(f"serving on http://127.0.0.1:{port} (waitress)")
        serve(app, host="127.0.0.1", port=port, threads=8)
    except ImportError:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

#GDTBATH
