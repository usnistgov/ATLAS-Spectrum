# ATLAS - Automatic Tiled Layering for Analyzing Spectra

"Geo Mapping for RF spectrum": one interface where **X is always time** and
colour is power. Zoom continuously from a two-year overview down to microsecond
structure — the viewer swaps resolution layers as you go.

![CBRS spectrum overview](docs/cbrs_overview.jpg)

Two independent data sources run on the same machinery:

- **CBRS SEA monitoring**: the [NIST NASCTN CBRS SEA](https://www.nist.gov/programs-projects/spectrum-monitoring-cbrs-band)
  dataset (10 sensors, 3.5 GHz band, ~2 years) as one continuous time axis.
- **IQ captures**: STFT spectrograms of individual [NIST I/Q recordings](https://www.nist.gov/ctl/spectrum-technology-and-research-division/applied-systems-metrology-group/iq-data-sets)
  (SigMF / TDMS), each with its own axes.

The real datasets are multi-GB and are **not** in this repo. A built-in
synthetic demo runs with no downloads.

# Quick start

**You need:** Python 3.10+, `git`, and a browser. Use whichever of `python` /
`python3` / `py` works on your machine.

**1. Get the code**

```bash
git clone https://github.com/jimmylu7/ATLAS.git
cd ATLAS
```

**2. Create a virtual environment, then install**

A virtual environment is required on most Linux and macOS systems (a bare `pip
install` there fails with "externally-managed-environment").

```bash
python -m venv .venv

source .venv/bin/activate             # macOS / Linux
.venv\Scripts\activate                # Windows PowerShell

pip install -r requirements.txt
```

**3. Build the demo data and check the install**

```bash
python atlas.py
```

With nothing built yet it offers to build a small synthetic capture
(`iq.duckdb`, ~2 MB) and start the viewer. Pick that; the check must end with
`RESULT: PASS`. If it says `FAIL`, the line above it names the problem.

**4. Start the viewer**

```bash
python atlas.py serve
```

Open **http://127.0.0.1:8090**, then pick **"IQ capture: demo_signal"** from the
**Source** dropdown.

![Synthetic demo spectrogram](docs/demo_spectrogram.png)

You should see a horizontal line (a carrier), stepping horizontal segments (a
hopping tone), and evenly spaced vertical bands (periodic bursts). Scroll to
zoom in time, `Ctrl`+scroll to zoom in frequency, drag to pan. `Ctrl+C` in the
terminal stops the server.

That is the whole install. Everything below is optional.

# CBRS data without the download

The quick start builds one IQ capture, so the CBRS sensor controls and the PSD
and PFP layers have nothing behind them. A synthetic stand-in builds all four
databases in seconds:

```bash
python examples/make_sample.py
python atlas.py serve
```

# Commands

| Command | What it does |
|---|---|
| `python atlas.py` | interactive menu: it checks the machine and offers the fix |
| `python atlas.py get <thing>` | a folder, file, dataset name, NIST record id, DOI or URL — it works out the rest |
| `python atlas.py doctor` | full report: environment, dependencies, disk, databases |
| `python atlas.py status` | what is built |
| `python atlas.py serve` | start the viewer |
| `python examples/test_all.py` | the offline test suites; exits 0 = all good |

# What you are looking at

In CBRS mode the viewer swaps between three stored-resolution layers
automatically, on either axis:

| Layer | Resolution | Shown when |
|-------|------------|------------|
| **Summary** | ~90 s sweeps, 18 channels | zoomed out on both axes |
| **PSD** | 2250 bins @ 80 kHz | time span ≤ 3 days **or** freq window ≤ 40 MHz |
| **PFP** | 560 pts across a 10 ms frame | freq window ≤ 12 MHz **and** time span ≤ 6 h |

The vertical slider on the left is frequency zoom, and marks where each deeper
layer takes over (blue for PSD, amber for PFP). The **Detail** readout in the
footer names the layer actually on screen. IQ mode renders each capture as its
own STFT pyramid on its own axes.

![IQ capture spectrogram](docs/iq_capture.jpg)

# Layout

```
atlas.py            one command: setup / get / status / serve
serve.py            Flask backend (app entry point)
viewer.html         single-file canvas frontend
datasets.json       friendly names -> PDR records
requirements.txt    dependencies, with tested versions
examples/           demo data, the install check, and the offline test suites
docs/               the manual, screenshots, IQ dataset notes
ingest/             fetch + database-build tooling, all resumable
```

`serve.py` renders WebP tiles from the coarsest stored level that still has
detail for the window, reading the DuckDB files read-only. Spectra are quantized
**uint8** in zlib-compressed chunks. This repo contains **only code**, no
spectrum data.

# Detailed manual

The [ATLAS manual](docs/README_MANUAL.md) contains the complete operational
reference: synthetic demo data, all `atlas.py` subcommands and flags,
troubleshooting, both viewer modes in full, `atlas.py get` and every download
option, bring-your-own datasets, environment variables, the ingest scripts, the
test suites, and architecture.

# AI Assistance

Parts of this project were built with the help of generative AI tools. All
AI-generated outputs were reviewed and tested by the human project maintainer. Claude Code and Claude Cowork were used to create code, compress data from the NIST PDR repositories and test the code once written. Behavior and fidelity of color mapping was checked by Jimmy Lu [jflu@unc.edu](mailto:jflu@unc.edu) and Aric Sanders [aric.sanders@nist.gov](mailto:aric.sanders@nist.gov). 

# Contact
## Primary Developer
Jimmy Lu
GitHub: [@jimmylu7](https://github.com/jimmylu7)
Email: [jflu@unc.edu](mailto:jflu@unc.edu) or [jimmy.lu@nist.gov](mailto:jimmy.lu@nist.gov)

## NIST Advisor
Aric Sanders
Email:[aric.sanders@nist.gov](mailto:aric.sanders@nist.gov)

# NIST data acknowledgment & disclaimer

This project visualizes public datasets from the National Institute of Standards
and Technology (NIST): the NASCTN CBRS SEA monitoring data and the Applied Systems
Metrology Group I/Q data sets. 

Certain commercial equipment, instruments, software, or materials may be
identified in this repository to foster understanding. Such identification does
not imply recommendation or endorsement by the National Institute of Standards
and Technology, nor does it imply that the materials or equipment identified are
necessarily the best available for the purpose.
