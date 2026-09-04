"""JEEG: a JData wrapper for electrophysiology recordings.

Built to the same shape as JNIfTI (:mod:`jdata.jnifti`): a parsed header
alongside the raw sample array, so that a document can carry the searchable
metadata inline while the bulk samples live in a compressed binary JData
attachment.

    {
      "EEGHeader":   {...},          # everything but the samples
      "EEGChannels": [ {...}, ... ], # per-channel calibration and units
      "EEGData":     ndarray,        # the samples
      "EEGEvents":   [ {...}, ... ]  # annotations, when the format carries them
    }

**The samples are stored as the recording's own integers, not as volts.**
Every format here records fixed-point integers plus a linear calibration, and
storing the integers keeps the file lossless, keeps it compressing well (a
float conversion would destroy the bit patterns deflate relies on), and leaves
the choice of scaling to the reader. ``EEGChannels[i]`` carries ``Gain`` and
``Offset`` such that

    physical = digital * Gain + Offset

so a consumer that wants microvolts applies two numbers per channel. For
formats that genuinely store floats, ``Gain`` is 1 and ``Offset`` 0 and the
array is float already.

Channels are stored as rows -- ``EEGData[channel, sample]`` -- which is the
layout every one of these formats is read in, and which keeps one channel
contiguous for the common case of looking at one electrode.

Supported inputs: EDF, EDF+, BDF (``.edf``/``.bdf``), BrainVision
(``.vhdr`` plus its ``.eeg``/``.dat``), and EEGLAB (``.set`` plus ``.fdt``).

author: Qianqian Fang <q.fang at neu.edu>
"""

import os
import re

import numpy as np

from .njdigest import bvheader, edfheader

__all__ = [
    "eeg2jeeg",
    "jeeg2edf",
    "edf2jeeg",
    "bv2jeeg",
    "eeglab2jeeg",
    "JEEG_EXTENSIONS",
]

#: input extensions this module can read
JEEG_EXTENSIONS = (".edf", ".bdf", ".vhdr", ".set")


def eeg2jeeg(filename, **kwargs):
    """Read any supported electrophysiology file into a JEEG structure.

    Dispatch is on the extension rather than on content sniffing, because the
    formats that matter here are identified by their sidecar naming anyway
    (a BrainVision ``.vhdr`` is meaningless without the ``.eeg`` it names).
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".edf", ".bdf"):
        return edf2jeeg(filename, **kwargs)

    if ext == ".vhdr":
        return bv2jeeg(filename, **kwargs)

    if ext == ".set":
        return eeglab2jeeg(filename, **kwargs)

    raise ValueError("no JEEG reader for %r" % ext)


# =============================================================================
# EDF / EDF+ / BDF
# =============================================================================


def edf2jeeg(filename, maxsamples=None):
    """Read an EDF, EDF+ or BDF recording.

    Signals in these formats may carry different sample counts per data
    record, which makes the natural layout ragged.  When they agree -- the
    overwhelmingly common case -- the samples come back as one 2-D array.
    When they do not, each channel comes back as its own 1-D array, because
    padding to a rectangle would invent samples that were never recorded.
    """
    with open(filename, "rb") as fid:
        magic = fid.read(8)

    # EyeLink and a few other tools also use the .edf extension for an entirely
    # different container; say so plainly instead of failing deep in a field parse
    if magic[:1] != b"\xff" and not magic[:8].strip().isdigit():
        raise ValueError(
            "%s is not an EDF/BDF recording (header starts %r); the extension is "
            "shared with unrelated formats such as EyeLink" % (filename, magic[:8])
        )

    meta = edfheader(filename)
    head = meta["EDFHeader"]
    subtype, continuity = _edf_subtype(head)

    nsig = head["NumberOfSignals"]
    source = _source_block(filename, subtype, 256 + 256 * nsig)
    perrec = head["SamplesPerRecord"]
    nrec = head["NumberOfDataRecords"]
    width = 3 if head["Format"] == "BDF" else 2
    offset = head["HeaderBytes"]

    if any(s is None for s in perrec) or nsig == 0:
        raise ValueError("unusable per-signal sample counts in %s" % filename)

    recsamples = int(sum(perrec))

    # A header may claim -1 records (unknown, allowed by EDF+); derive the
    # count from the file size instead of trusting it.
    filesize = os.path.getsize(filename)
    available = max(0, filesize - offset) // (recsamples * width) if recsamples else 0

    if nrec is None or nrec < 0 or nrec > available:
        nrec = int(available)

    if maxsamples:
        cap = max(1, int(maxsamples) // max(1, recsamples))
        nrec = min(nrec, cap)

    raw = np.fromfile(filename, dtype=np.uint8, offset=offset, count=nrec * recsamples * width)
    usable = (len(raw) // (recsamples * width)) if recsamples else 0

    if usable < nrec:
        nrec = int(usable)
        raw = raw[: nrec * recsamples * width]

    values = _decode_fixed_width(raw, width)
    values = values.reshape(nrec, recsamples) if nrec else values.reshape(0, recsamples)

    # slice each signal out of every record, then concatenate along time
    bounds = np.cumsum([0] + list(perrec))
    channels = [values[:, bounds[i] : bounds[i + 1]].reshape(-1) for i in range(nsig)]

    # EDF+ hides its annotations in a signal named "EDF Annotations", whose
    # "samples" are text, not measurements. Reading it as a data channel mixes
    # TAL bytes into the sample matrix and loses the events entirely.
    labels = head.get("Labels") or []
    annot_idx = [
        i
        for i in range(nsig)
        if str(labels[i] if i < len(labels) else "").strip()
        in ("EDF Annotations", "BDF Annotations")
    ]
    signal_idx = [i for i in range(nsig) if i not in annot_idx]

    events, starts = [], []

    if annot_idx and nrec:
        events, starts = _edf_annotations(
            raw, bounds, perrec, recsamples, width, nrec, annot_idx
        )

    kept = [channels[i] for i in signal_idx]
    kept_lengths = [perrec[i] for i in signal_idx]
    uniform = len(set(kept_lengths)) == 1
    data = np.vstack(kept) if (uniform and kept) else kept

    allchan = _edf_channels(head)

    for i, ch in enumerate(allchan):
        ch["OriginalIndex"] = i

    out = {
        "EEGHeader": _edf_header(head, nrec, subtype, continuity, len(annot_idx)),
        "EEGChannels": [allchan[i] for i in signal_idx],
        "EEGData": data,
        "EEGSource": source,
    }

    if events:
        out["EEGEvents"] = events

    # The annotation signal is kept verbatim as well as parsed. EEGEvents is
    # what a reader wants; this is what a writer needs, because the samples of
    # that channel are the file's own bytes and no parse of them round-trips.
    if annot_idx:
        out["EEGAnnotationData"] = {
            "OriginalIndex": annot_idx,
            "SamplesPerRecord": [perrec[i] for i in annot_idx],
            "Data": np.vstack([channels[i] for i in annot_idx])
            if len(set(perrec[i] for i in annot_idx)) == 1
            else [channels[i] for i in annot_idx],
        }

    # EDF+D is not a continuous recording: the records carry their own start
    # times and there are gaps between them, so a reader that assumes a uniform
    # time axis gets it wrong
    if continuity == "discontinuous" and starts:
        out["EEGHeader"]["RecordStartTimes"] = starts

    return out




def _source_block(filename, fmt, headerbytes=0):
    """What is needed to reconstruct, and to prove a reconstruction correct.

    Encoding does not keep the original bytes in the content store, so the
    wrapper is the only remaining description of the file. Two things make it
    sufficient: the original header captured verbatim -- these formats write
    fixed-width ASCII whose exact spelling ("1000" against "1.0E3") no parse
    preserves -- and the size and digest of the whole file, so a rebuild can be
    checked rather than assumed.
    """
    import hashlib

    out = {
        "Format": fmt,
        "File": os.path.basename(filename),
        "Bytes": os.path.getsize(filename),
    }

    digest = hashlib.sha256()

    with open(filename, "rb") as fid:
        if headerbytes:
            raw = fid.read(headerbytes)
            out["RawHeader"] = {"_ByteStream_": raw}
            digest.update(raw)

        for chunk in iter(lambda: fid.read(1 << 20), b""):
            digest.update(chunk)

    out["SHA256"] = digest.hexdigest()
    return out


def _edf_subtype(head):
    """Tell EDF from EDF+ (and BDF from BDF+), and continuous from not.

    The + variants are marked only in the 44-byte reserved field: "EDF+C" for
    continuous, "EDF+D" for discontinuous. Plain BDF writes "24BIT" there and
    plain EDF usually leaves it blank, so the base format still comes from the
    version bytes.
    """
    base = head.get("Format") or "EDF"
    reserved = (head.get("Reserved") or "").strip().upper()

    if reserved.startswith(("EDF+", "BDF+")):
        return base + "+", ("discontinuous" if reserved.endswith("D") else "continuous")

    return base, "continuous"


def _edf_annotations(raw, bounds, perrec, recsamples, width, nrec, annot_idx):
    """Parse EDF+ TALs out of the annotation signal.

    Each record holds one or more Time-stamped Annotation Lists:

        +<onset>[\x15<duration>]\x14<text>[\x14<text>...]\x14\x00

    The first TAL of a record carries an empty text and states that record's
    start time, which is what makes EDF+D navigable.
    """
    events, starts = [], []
    recbytes = recsamples * width

    for r in range(nrec):
        for i in annot_idx:
            lo = r * recbytes + bounds[i] * width
            hi = lo + perrec[i] * width
            blob = bytes(raw[lo:hi])

            for n, tal in enumerate(blob.split(b"\x00")):
                if not tal.strip(b"\x00 \t"):
                    continue

                parsed = _parse_tal(tal)

                if parsed is None:
                    continue

                onset, duration, texts = parsed

                if n == 0 and not any(t for t in texts):
                    starts.append(onset)   # the record-start TAL
                    continue

                for text in texts:
                    if not text:
                        continue

                    entry = {"Onset": onset, "Annotation": text}

                    if duration is not None:
                        entry["Duration"] = duration

                    events.append(entry)

    return events, starts


def _parse_tal(tal):
    """One TAL: onset, optional duration, then one or more \x14-joined texts."""
    try:
        body = tal.decode("utf-8", "replace")
    except Exception:
        return None

    parts = body.split("\x14")

    if not parts or not parts[0]:
        return None

    stamp = parts[0]
    duration = None

    if "\x15" in stamp:
        stamp, _, dur = stamp.partition("\x15")

        try:
            duration = float(dur)
        except ValueError:
            duration = None

    try:
        onset = float(stamp)
    except ValueError:
        return None

    return onset, duration, [p.strip("\x00").strip() for p in parts[1:]]


def _decode_fixed_width(raw, width):
    """Little-endian two's-complement integers of 2 or 3 bytes each."""
    if width == 2:
        return raw.view("<i2").astype(np.int32)

    # BDF stores 24-bit samples; sign-extend into int32
    if len(raw) % 3:
        raw = raw[: len(raw) - (len(raw) % 3)]

    triples = raw.reshape(-1, 3).astype(np.int32)
    out = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    return np.where(out >= (1 << 23), out - (1 << 24), out).astype(np.int32)


def _edf_header(head, nrec, subtype=None, continuity=None, nannot=0):
    # every field the header parser produced, minus the per-signal lists that
    # are reported per channel in EEGChannels instead. Dropping anything else
    # would make the original file unreconstructable.
    perchannel = (
        "Labels",
        "TransducerType",
        "PhysicalDimension",
        "PhysicalMinimum",
        "PhysicalMaximum",
        "DigitalMinimum",
        "DigitalMaximum",
        "PreFiltering",
        "SamplesPerRecord",
        "SamplingFrequency",
    )
    out = {k: v for k, v in head.items() if k not in perchannel}

    if subtype:
        out["Format"] = subtype

    if continuity:
        out["Continuity"] = continuity

    # NumberOfSignals in the file counts the annotation channel; the document
    # should describe the channels it actually carries
    if nannot:
        out["NumberOfSignals"] = max(0, int(head.get("NumberOfSignals") or 0) - nannot)
        out["NumberOfAnnotationSignals"] = nannot

    out["NumberOfDataRecords"] = nrec
    out["RecordingDuration"] = (
        nrec * head["DurationOfDataRecord"] if head.get("DurationOfDataRecord") else None
    )
    out["DataLayout"] = "channel-major"

    # one figure for the whole recording when the channels agree, which is the
    # common case; per-channel rates stay in EEGChannels regardless
    rates = [r for r in (head.get("SamplingFrequency") or []) if r]
    if rates and len(set(rates)) == 1:
        out["SamplingFrequency"] = rates[0]

    return out


def _edf_channels(head):
    """Per-channel calibration: physical = digital * Gain + Offset."""
    out = []

    for i in range(head["NumberOfSignals"]):

        def at(key, default=None):
            seq = head.get(key) or []
            return seq[i] if i < len(seq) else default

        pmin, pmax = at("PhysicalMinimum"), at("PhysicalMaximum")
        dmin, dmax = at("DigitalMinimum"), at("DigitalMaximum")
        gain, offset = 1.0, 0.0

        if None not in (pmin, pmax, dmin, dmax) and dmax != dmin:
            gain = (pmax - pmin) / float(dmax - dmin)
            offset = pmin - dmin * gain

        out.append(
            {
                "Label": at("Labels", ""),
                "Unit": at("PhysicalDimension", ""),
                "SamplingFrequency": at("SamplingFrequency"),
                "SamplesPerRecord": at("SamplesPerRecord"),
                "Gain": gain,
                "Offset": offset,
                "PhysicalMinimum": pmin,
                "PhysicalMaximum": pmax,
                "DigitalMinimum": dmin,
                "DigitalMaximum": dmax,
                "TransducerType": at("TransducerType", ""),
                "PreFiltering": at("PreFiltering", ""),
            }
        )

    return out


# =============================================================================
# BrainVision
# =============================================================================

_BV_DTYPE = {
    "int_16": "<i2",
    "int_32": "<i4",
    "ieee_float_32": "<f4",
    "ieee_float_64": "<f8",
    "uint_16": "<u2",
}


def bv2jeeg(filename, maxsamples=None):
    """Read a BrainVision recording named by its ``.vhdr``.

    The header is an INI file naming the binary beside it. Only the binary
    layouts are handled -- multiplexed and vectorized -- which is what BIDS
    EEG and iEEG datasets contain in practice.
    """
    common, binary, chaninfo = _bv_sections(bvheader(filename))

    nchan = int(common.get("NumberOfChannels") or 0)
    fmt = str(binary.get("BinaryFormat") or "INT_16").lower()
    orient = str(common.get("DataOrientation") or "MULTIPLEXED").upper()
    interval = float(common.get("SamplingInterval") or 0)
    dataformat = str(common.get("DataFormat") or "BINARY").upper()

    if nchan <= 0:
        raise ValueError("BrainVision header names no channels: %s" % filename)

    if dataformat != "BINARY":
        raise ValueError("only BINARY BrainVision data is supported, not %r" % dataformat)

    dtype = _BV_DTYPE.get(fmt)

    if dtype is None:
        raise ValueError("unsupported BrainVision BinaryFormat %r" % fmt)

    binpath = _bv_binary(filename, common.get("DataFile"))
    count = -1 if not maxsamples else int(maxsamples)
    raw = np.fromfile(binpath, dtype=dtype, count=count)

    nsamp = len(raw) // nchan
    raw = raw[: nsamp * nchan]
    # MULTIPLEXED interleaves channels per sample; VECTORIZED stores each
    # channel end to end
    data = (
        raw.reshape(nsamp, nchan).T
        if orient.startswith("MULTI")
        else raw.reshape(nchan, nsamp)
    )
    srate = (1e6 / interval) if interval else None

    return {
        "EEGHeader": {
            "Format": "BrainVision",
            "DataOrientation": orient,
            "BinaryFormat": fmt,
            "NumberOfChannels": nchan,
            "NumberOfSamples": int(nsamp),
            "SamplingInterval": interval or None,
            "SamplingFrequency": srate,
            "RecordingDuration": (nsamp * interval / 1e6) if interval else None,
            "DataFile": os.path.basename(binpath),
            "MarkerFile": common.get("MarkerFile"),
            "DataLayout": "channel-major",
        },
        "EEGChannels": _bv_channels(chaninfo, nchan, srate),
        "EEGData": np.ascontiguousarray(data),
        # the .vhdr is small ASCII, so it is kept whole rather than described;
        # with it plus BinaryFormat and DataOrientation the pair is rebuildable
        "EEGSource": _source_block(filename, "BrainVision", os.path.getsize(filename)),
        "EEGSourceBinary": _source_block(binpath, "BrainVision-binary"),
    }


def _bv_sections(meta):
    """bvheader returns {"VHDRHeader": {"CommonInfos": .., "BinaryInfos": .., ..}}."""
    head = meta.get("VHDRHeader") or meta.get("BrainVisionHeader") or meta or {}
    return (
        head.get("CommonInfos") or {},
        head.get("BinaryInfos") or {},
        head.get("ChannelInfos") or {},
    )


def _bv_binary(vhdr, named):
    """Resolve the binary companion, falling back on the usual extensions."""
    base = os.path.dirname(os.path.abspath(vhdr))
    stem = os.path.splitext(os.path.basename(vhdr))[0]
    candidates = []

    if named:
        candidates.append(os.path.join(base, os.path.basename(str(named))))

    candidates += [os.path.join(base, stem + e) for e in (".eeg", ".dat", ".EEG", ".DAT")]

    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.realpath(path)):
            return path

    raise ValueError("BrainVision binary not found for %s" % vhdr)


def _bv_channels(chaninfo, nchan, srate):
    """ChannelInfos arrives already split into parallel lists."""
    names = list(chaninfo.get("name") or [])
    units = list(chaninfo.get("unit") or [])
    res = list(chaninfo.get("resolution") or [])
    refs = list(chaninfo.get("reference") or [])
    out = []

    for i in range(nchan):
        gain = res[i] if i < len(res) and res[i] else 1.0
        out.append(
            {
                "Label": names[i] if i < len(names) else "",
                # BrainVision leaves the unit blank when it means microvolts
                "Unit": (units[i] if i < len(units) and units[i] else "uV"),
                "Reference": refs[i] if i < len(refs) else "",
                "SamplingFrequency": srate,
                "Gain": float(gain),
                "Offset": 0.0,
                "Resolution": (float(res[i]) if i < len(res) and res[i] is not None else None),
            }
        )

    return out


# =============================================================================
# EEGLAB
# =============================================================================


def eeglab2jeeg(filename, maxsamples=None):
    """Read an EEGLAB ``.set``.

    ``EEG`` arrives from scipy as a 1x1 structured array whose every field is
    itself an array, so each value needs unwrapping twice. The samples live
    either inside the MAT-file as ``EEG.data`` or -- for anything large, which
    is the norm -- in a separate ``.fdt`` that ``EEG.data`` names as a string.
    Epoched recordings are stored channels x points x trials and are flattened
    to channels x (points*trials), keeping the trial count in the header so the
    shape can be recovered.
    """
    from .jfile import loadmat

    raw = loadmat(filename)
    eeg = raw.get("EEG", raw)

    if hasattr(eeg, "dtype") and getattr(eeg.dtype, "names", None):
        rec = eeg.ravel()[0] if eeg.size else None
        names = set(eeg.dtype.names)

        def get(key, default=None):
            return rec[key] if (rec is not None and key in names) else default

    elif isinstance(eeg, dict):
        get = eeg.get
    else:
        raise ValueError("unrecognised EEGLAB structure in %s" % filename)

    nchan = _scalar(get("nbchan"), 0, int) or 0
    npoint = _scalar(get("pnts"), 0, int) or 0
    ntrial = _scalar(get("trials"), 1, int) or 1
    srate = _scalar(get("srate"), None, float)
    data = get("data")

    external = None

    if data is not None:
        arr = np.asarray(data)

        if arr.dtype.kind in "US" or arr.dtype == object and arr.size == 1:
            candidate = arr.ravel()[0] if arr.size else None

            if isinstance(candidate, (str, bytes, np.str_)):
                external = str(candidate)

    if external:
        fdt = os.path.join(os.path.dirname(os.path.abspath(filename)), os.path.basename(external))

        if not (os.path.exists(fdt) and os.path.exists(os.path.realpath(fdt))):
            raise ValueError("EEGLAB .fdt companion not available: %s" % external)

        count = -1 if not maxsamples else int(maxsamples)
        flat = np.fromfile(fdt, dtype="<f4", count=count)

        if nchan <= 0:
            raise ValueError("EEGLAB .set names no channels: %s" % filename)

        # the .fdt is written channel-fastest, i.e. one sample's channels
        # together, so it reshapes as (samples, channels) then transposes
        usable = len(flat) // nchan
        samples = np.ascontiguousarray(flat[: nchan * usable].reshape(usable, nchan).T)
    else:
        samples = np.asarray(data)

        if samples.ndim == 3:
            samples = samples.reshape(samples.shape[0], -1)

    labels = _eeglab_labels(get("chanlocs"), nchan)
    nsamp = int(samples.shape[1]) if getattr(samples, "ndim", 0) == 2 else None
    fdtsource = _source_block(fdt, "EEGLAB-fdt") if external else None

    return {
        "EEGHeader": {
            "Format": "EEGLAB",
            "SetName": _text(get("setname")),
            "NumberOfChannels": nchan or (int(samples.shape[0]) if samples.ndim == 2 else 0),
            "NumberOfSamples": nsamp,
            "PointsPerTrial": npoint or None,
            "NumberOfTrials": ntrial,
            "SamplingFrequency": srate,
            "RecordingDuration": (nsamp / srate) if (nsamp and srate) else None,
            "DataFile": os.path.basename(external) if external else None,
            "DataLayout": "channel-major",
        },
        "EEGChannels": [
            {
                "Label": labels[i] if i < len(labels) else "",
                "Unit": "uV",
                "SamplingFrequency": srate,
                "Gain": 1.0,
                "Offset": 0.0,
            }
            for i in range(nchan or 0)
        ],
        "EEGData": samples,
        "EEGSource": _source_block(filename, "EEGLAB"),
        **({"EEGSourceBinary": fdtsource} if fdtsource else {}),
    }


def _eeglab_labels(chanlocs, nchan):
    """chanlocs is its own structured array; pull the ``labels`` field."""
    if chanlocs is None:
        return []

    arr = np.asarray(chanlocs)

    if getattr(arr.dtype, "names", None) and "labels" in arr.dtype.names:
        flat = arr.ravel()
        return [_text(flat[i]["labels"]) for i in range(min(len(flat), nchan or len(flat)))]

    out = []

    for entry in arr.ravel()[: nchan or None]:
        if isinstance(entry, dict):
            out.append(_text(entry.get("labels")))
        else:
            out.append("")

    return out


def _scalar(value, default=None, cast=None):
    if value is None:
        return default

    try:
        flat = np.asarray(value).ravel()

        if not flat.size:
            return default

        out = flat[0]
    except Exception:
        out = value

    if cast is not None:
        try:
            return cast(out)
        except (TypeError, ValueError):
            return default

    return out


def _text(value):
    if value is None:
        return ""

    try:
        flat = np.asarray(value).ravel()
        return str(flat[0]) if flat.size else ""
    except Exception:
        return str(value)


# =============================================================================
# reconstruction
# =============================================================================


def jeeg2edf(jeeg, filename):
    """Write an EDF/EDF+/BDF back out from its JEEG structure.

    Uses ``EEGSource.RawHeader`` verbatim, so the header is reproduced exactly
    rather than re-derived from parsed fields whose spelling was never
    preserved.  The data section is rebuilt by re-interleaving the channels
    into records, annotation channels included at their original positions.

    Returns the SHA-256 of what was written, which
    ``EEGSource.SHA256`` can be checked against.
    """
    import hashlib

    source = jeeg.get("EEGSource") or {}
    rawhdr = (source.get("RawHeader") or {}).get("_ByteStream_")

    if not rawhdr:
        raise ValueError("this JEEG has no EEGSource.RawHeader; cannot rebuild exactly")

    head = jeeg["EEGHeader"]
    width = 3 if str(head.get("Format", "")).startswith("BDF") else 2
    nrec = int(head["NumberOfDataRecords"])

    # put every channel back at its original index
    slots = {}

    for ch, row in zip(jeeg.get("EEGChannels") or [], np.atleast_2d(jeeg["EEGData"])):
        slots[int(ch["OriginalIndex"])] = (np.asarray(row), int(ch["SamplesPerRecord"]))

    annot = jeeg.get("EEGAnnotationData")

    if annot:
        rows = np.atleast_2d(annot["Data"])

        for idx, per, row in zip(annot["OriginalIndex"], annot["SamplesPerRecord"], rows):
            slots[int(idx)] = (np.asarray(row), int(per))

    order = sorted(slots)
    body = bytearray()

    for r in range(nrec):
        for i in order:
            row, per = slots[i]
            chunk = row[r * per : (r + 1) * per].astype(np.int64)

            if width == 2:
                body += chunk.astype("<i2").tobytes()
            else:
                for v in chunk:
                    body += (int(v) & 0xFFFFFF).to_bytes(3, "little")

    payload = bytes(rawhdr) + bytes(body)

    with open(filename, "wb") as fid:
        fid.write(payload)

    return hashlib.sha256(payload).hexdigest()
