# NIST I/Q datasets: format notes for iq_ingest.py

Ingest: `py iq_ingest.py <folder> --dataset <name>` → iq.duckdb (STFT pyramid,
uint8, fixed per-capture qmin/qmax). Formats handled by sigmf_io.open_capture:
SigMF (.sigmf-meta/.sigmf-data), NI TDMS (.tdms), .npy. Special cases found by
probing the PDR records (data.nist.gov/rmm/records/<id>):

- **mds2-3177** (High-SNR FDD LTE, COTS handset): DONE end-to-end. Not SigMF,
  it's NI TDMS. 'Header Data' group carries carrier_frequency (1.77 GHz),
  IQ_samples_per_second (1.92/3.84/7.68/15.36 Msps by bandwidth folder),
  reference_level_dBm (int16 → volts per the README formula, so tiles are true
  dBm @ 50 ohm). 'IQ Signal' group: int16 'I Data'/'Q Data'. 38-307 MB/file.
  test_config_*.csv maps config number → modulation/RBs.
- **mds2-2413** (field LTE via telemetry antenna): files are 1.2-3 GB **zips**
  (SingleUE/, MultiUE - Day 1/2). Unzip first, then point iq_ingest at the
  extracted folder; contents expected TDMS-like (same instrument family).
- **mds2-2395** (lab LTE uplink): 0.8-1.2 GB **zips** per
  IQ_Library_Configuration_*/Capture_*/Data.zip. Same: unzip, then ingest.
- **mds2-2731** (Wi-Fi/Bluetooth 2.4/5 GHz): five **21.6 GB HDF5** (.h5)
  files, not SigMF. Needs an HDF5 adapter in sigmf_io (h5py, dataset layout
  TBD from README.txt) plus chunked reads; do NOT try to download casually.

Gotcha reminders: DOIs 302-redirect to JS SPA landing pages, so use the PDR
record API for file lists/URLs. Never hardcode fs/fc/datatype; read them from
each capture's metadata. WAL files next to iq.duckdb from an interrupted
ingest should be removed before serving (stale-WAL replay).
