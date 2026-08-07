"""
sigmf_io.py: unified reader for IQ capture files (SigMF and friends).

Gives iq_ingest.py / serve.py one interface over the formats the NIST
"I/Q Data Sets" actually ship:
    .sigmf-meta / .sigmf-data   SigMF (JSON sidecar + raw interleaved I/Q)
    .tdms                       NI LabVIEW (mds2-3177 High-SNR LTE uplink)
    .complex1ch.float32         Rohde & Schwarz iq-tar, samples + .xml sidecar
                                (mds2-3226 microwave oven, mds2-3684 TIG welding)
    .npy / .mat                 fallback for datasets shipping numpy/MATLAB

open_capture(path) -> IQCapture with:
    sample_rate (Hz), center_freq (Hz), datatype, n_samples, duration (s),
    annotations (list), name
    read(start, end) -> complex64 ndarray for [start, end) samples
Readers memory-map / lazy-read: a whole capture is NEVER loaded into RAM.
"""

import json
import os

import numpy as np

# SigMF core:datatype -> (numpy scalar dtype, is_complex_interleaved, scale)
# cf32 = complex float32; ci16 = interleaved int16 I/Q; cu8 etc.
_SIGMF_DTYPES = {
    "cf64_le": ("<c16", False, 1.0), "cf64_be": (">c16", False, 1.0),
    "cf32_le": ("<c8", False, 1.0),  "cf32_be": (">c8", False, 1.0),
    "ci32_le": ("<i4", True, 1 / 2**31), "ci32_be": (">i4", True, 1 / 2**31),
    "ci16_le": ("<i2", True, 1 / 2**15), "ci16_be": (">i2", True, 1 / 2**15),
    "ci8":  ("i1", True, 1 / 2**7), "cu8": ("u1", True, 1 / 2**7),
}


class IQCapture:
    """Base: metadata + slice reader. Subclasses implement read()."""

    def __init__(self, path, name, sample_rate, center_freq, datatype,
                 n_samples, annotations=None):
        if sample_rate is None or sample_rate <= 0:
            raise ValueError(f"{path}: missing/invalid sample_rate")
        if n_samples <= 0:
            raise ValueError(f"{path}: no samples")
        self.path = path
        self.name = name
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq or 0.0)
        self.datatype = datatype
        self.n_samples = int(n_samples)
        self.duration = self.n_samples / self.sample_rate
        self.annotations = annotations or []

    def read(self, start, end):
        raise NotImplementedError


class SigMFCapture(IQCapture):
    """SigMF: .sigmf-meta JSON sidecar + memory-mapped .sigmf-data binary."""

    def __init__(self, meta_path):
        # SigMF metadata is UTF-8 by spec. open() without an explicit encoding
        # uses the locale codec, which is still cp1252 on Windows, so a
        # sidecar containing "us" or a degree sign dies with a
        # UnicodeDecodeError there while working fine on Linux.
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        g = meta.get("global", {})
        dt = g.get("core:datatype")
        if dt not in _SIGMF_DTYPES:
            raise ValueError(f"{meta_path}: unknown SigMF datatype {dt!r}")
        self._np_dtype, self._interleaved, self._scale = _SIGMF_DTYPES[dt]
        caps = meta.get("captures", [{}])
        fc = caps[0].get("core:frequency")
        if fc is None:  # some NTIA-extension records put it elsewhere
            for k, v in g.items():
                if k.endswith(":center_frequency") or k.endswith(":frequency"):
                    fc = v
                    break
        data_path = meta_path[:-len(".sigmf-meta")] + ".sigmf-data"
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        itemsize = np.dtype(self._np_dtype).itemsize * (2 if self._interleaved else 1)
        fsize = os.path.getsize(data_path)
        if fsize % itemsize:
            raise ValueError(f"{data_path}: size {fsize} not a multiple of "
                             f"sample size {itemsize} ({dt})")
        n = fsize // itemsize
        # An interrupted download leaves a .sigmf-data that is simply shorter,
        # and nothing above notices: n is derived from the file's own size, so
        # duration, column count and the time axis are all recomputed from the
        # short length and the capture ingests looking perfectly healthy. The
        # sidecar is the only record of how long it should have been --
        # annotations mark spans of real samples, so one running past the end
        # of the file means samples are missing. Cheap, and it turns a silent
        # 90%-complete capture into a plain "re-download this" instead.
        declared = 0
        for a in (meta.get("annotations") or []):
            s, c = a.get("core:sample_start"), a.get("core:sample_count")
            if isinstance(s, (int, float)) and isinstance(c, (int, float)):
                declared = max(declared, int(s + c))
        if declared > n:
            raise ValueError(
                f"{data_path}: truncated -- the metadata describes {declared} "
                f"samples but the file holds {n} ({100.0 * n / declared:.1f}%). "
                "Re-download this capture.")
        self._mm = np.memmap(data_path, dtype=self._np_dtype, mode="r")
        super().__init__(meta_path, os.path.basename(meta_path)[:-len(".sigmf-meta")],
                         g.get("core:sample_rate"), fc, dt, n,
                         meta.get("annotations", []))

    def read(self, start, end):
        if self._interleaved:
            raw = self._mm[2 * start:2 * end].astype(np.float32) * self._scale
            return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        return np.asarray(self._mm[start:end], dtype=np.complex64)


class TdmsCapture(IQCapture):
    """NI TDMS as shipped by NIST mds2-3177: 'Header Data' group holds
    reference_level_dBm + IQ samples per second; 'IQ Signal' group holds
    int16 'I Data'/'Q Data' channels. Values scale to volts via the header's
    reference level (per the dataset README), so power is real dBm @ 50 ohm."""

    def __init__(self, path, center_freq=None):
        from nptdms import TdmsFile
        self._tf = TdmsFile.open(path)   # lazy: channels stream from disk
        hdr = {}
        for grp in self._tf.groups():
            if grp.name.lower().startswith("header"):
                for ch in grp.channels():
                    v = ch[:]
                    hdr[ch.name.strip().lower().replace(" ", "_")] = (
                        v[0] if len(v) else None)
        sig = None
        for grp in self._tf.groups():
            names = {c.name.strip().lower(): c for c in grp.channels()}
            if "i data" in names and "q data" in names:
                sig = (names["i data"], names["q data"])
                break
        if sig is None:
            raise ValueError(f"{path}: no 'I Data'/'Q Data' channels")
        self._i, self._q = sig
        fs = hdr.get("iq_samples_per_second")
        ref_dbm = hdr.get("reference_level_dbm")
        # per README: volts-per-bit = sqrt(10^(ref/10)*Z)/2^16   (Z = 50 ohm)
        self._scale = (np.sqrt(10 ** (float(ref_dbm) / 10) * 50.0) / 2**16
                       if ref_dbm is not None else 1 / 2**15)
        fc = center_freq
        if fc is None:
            for k in ("carrier_frequency", "center_frequency", "frequency",
                      "carrier_frequency_hz", "lo_frequency"):
                if hdr.get(k) is not None:
                    fc = float(hdr[k])
                    break
        super().__init__(path, os.path.splitext(os.path.basename(path))[0],
                         fs, fc, "tdms_i16", len(self._i),
                         [{"header": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                                      for k, v in hdr.items() if v is not None}}])

    def read(self, start, end):
        i = np.asarray(self._i[start:end], dtype=np.float32)
        q = np.asarray(self._q[start:end], dtype=np.float32)
        return ((i + 1j * q) * self._scale).astype(np.complex64)


class IqTarCapture(IQCapture):
    """Rohde & Schwarz iq-tar, as shipped extracted by NIST mds2-3226/mds2-3684.

    Each capture is a `<name>.iq/` directory holding one
    `File_<timestamp>.complex1ch.float32` of interleaved complex float32 beside
    the `.xml` that describes it. The XML is the only place the sample rate and
    centre frequency exist, so it is required; without it the file is an
    undifferentiated wall of floats.

    Relevant elements, as written by the instrument (an FSW-8 here):
        <Samples>50000000</Samples>
        <Clock unit="Hz">625000000.000000</Clock>
        <Format>complex</Format>  <DataType>float32</DataType>
        <ScalingFactor unit="V">1</ScalingFactor>
        <NumberOfChannels>1</NumberOfChannels>
        <CenterFrequency unit="Hz">900000000.000000</CenterFrequency>
    The file also ends with a long <PreviewData> block of <float> elements --
    an instrument-drawn thumbnail, not samples, and ignored here.
    """

    SUFFIX = ".complex1ch.float32"

    def __init__(self, path):
        import xml.etree.ElementTree as ET
        data_path, xml_path = self._pair(path)
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            raise ValueError(f"{xml_path}: not readable as iq-tar XML ({e})")

        def first(tag, default=None):
            for el in root.iter(tag):
                return (el.text or "").strip()
            return default

        fmt = (first("Format") or "").lower()
        dtype = (first("DataType") or "").lower()
        nch = first("NumberOfChannels", "1")
        if fmt != "complex" or dtype != "float32":
            raise ValueError(
                f"{xml_path}: this reader handles complex/float32 iq-tar, not "
                f"{fmt or '?'}/{dtype or '?'}")
        if nch not in ("1", None, ""):
            raise ValueError(f"{xml_path}: {nch} channels; only single-channel "
                             "iq-tar captures are supported")
        fs = first("Clock")
        if not fs:
            raise ValueError(f"{xml_path}: no <Clock>, so the sample rate is "
                             "unknown")
        declared = int(float(first("Samples") or 0))
        scale = float(first("ScalingFactor") or 1.0)

        # One complex sample is two float32s. A part-downloaded file is simply
        # shorter and nothing downstream would notice -- duration and the time
        # axis are both derived from the length -- so compare against the count
        # the instrument recorded and say so plainly instead.
        itemsize = 2 * np.dtype(np.float32).itemsize
        fsize = os.path.getsize(data_path)
        if fsize % itemsize:
            raise ValueError(f"{data_path}: size {fsize} is not a whole number "
                             f"of complex float32 samples")
        n = fsize // itemsize
        if declared and n < declared:
            raise ValueError(
                f"{data_path}: truncated -- the XML declares {declared} "
                f"samples but the file holds {n} ({100.0 * n / declared:.1f}%). "
                "Re-download this capture.")
        self._mm = np.memmap(data_path, dtype="<f4", mode="r")
        self._scale = scale
        hdr = {"instrument": first("Name"), "recorded": first("DateTime"),
               "bandwidth_hz": None}
        for el in root.iter("Key"):
            if (el.get("name") or "").endswith("MeasBandwidth[Hz]"):
                hdr["bandwidth_hz"] = (el.text or "").strip()
                break
        super().__init__(data_path, os.path.basename(os.path.dirname(data_path))
                         or os.path.basename(data_path),
                         float(fs), float(first("CenterFrequency") or 0.0),
                         "iqtar_cf32", n, [{"iqtar": hdr}])

    @classmethod
    def _pair(cls, path):
        """-> (samples file, xml file), given either one."""
        if path.lower().endswith(".xml"):
            data = path[:-len(".xml")]
            if not os.path.exists(data):
                raise FileNotFoundError(
                    f"{data}: the samples file this XML describes is missing")
            return data, path
        xml = path + ".xml"
        if not os.path.exists(xml):
            raise FileNotFoundError(
                f"{xml}: iq-tar captures need the .xml beside the samples for "
                "the sample rate and centre frequency")
        return path, xml

    def read(self, start, end):
        raw = np.asarray(self._mm[2 * start:2 * end], dtype=np.float32)
        if self._scale != 1.0:
            raw = raw * self._scale
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


class ArrayCapture(IQCapture):
    """Fallback for .npy (complex array), memory-mapped."""

    def __init__(self, path, sample_rate, center_freq=0.0):
        arr = np.load(path, mmap_mode="r")
        if not np.iscomplexobj(arr):
            raise ValueError(f"{path}: expected a complex array")
        self._mm = arr.reshape(-1)
        super().__init__(path, os.path.splitext(os.path.basename(path))[0],
                         sample_rate, center_freq, str(arr.dtype), self._mm.size)

    def read(self, start, end):
        return np.asarray(self._mm[start:end], dtype=np.complex64)


def open_capture(path, sample_rate=None, center_freq=None):
    """Detect format by extension and return an IQCapture."""
    low = path.lower()
    if low.endswith(".sigmf-meta"):
        return SigMFCapture(path)
    if low.endswith(".sigmf-data"):
        return SigMFCapture(path[:-len(".sigmf-data")] + ".sigmf-meta")
    if low.endswith(".tdms"):
        return TdmsCapture(path, center_freq=center_freq)
    if low.endswith(IqTarCapture.SUFFIX) or low.endswith(
            IqTarCapture.SUFFIX + ".xml"):
        return IqTarCapture(path)
    if low.endswith(".npy"):
        return ArrayCapture(path, sample_rate, center_freq or 0.0)
    if low.endswith(".mat"):
        raise ValueError(f"{path}: .mat needs a per-dataset adapter "
                         "(variable names differ). Add one in sigmf_io.py")
    raise ValueError(f"{path}: unsupported capture format")
