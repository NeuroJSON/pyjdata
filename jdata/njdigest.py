"""
Header-only ("digest") readers for bulky neuroimaging container formats.

A NeuroJSON digest must stay small enough to live in a database document while
remaining useful to search, so for large recordings only the *header* is parsed:
sampling rate, channel labels, units, dimensions, acquisition metadata.  The
sample data itself is referenced through a content-addressed ``_DataLink_``.

Every reader here is deliberately streaming or seek-based -- none of them loads
the sample payload -- so extracting metadata from a multi-gigabyte recording
costs a few kilobytes of I/O.

Formats covered:

======================  ==============================================
``.snirf`` ``.nwb``     via :func:`hdf5_digest` (structure + attributes)
``.h5`` ``.hdf5``       via :func:`hdf5_digest`
``.edf`` ``.bdf``       via :func:`edfheader` (EDF/EDF+/BDF)
``.vhdr`` ``.vmrk``     via :func:`bvheader` (BrainVision)
``.set``                via :func:`eeglab_digest` (HDF5 or MAT v5)
``.mat`` (FSL VEST)     via :func:`vest_header` (a text design matrix)
======================  ==============================================

EDF, BrainVision and EEGLAB metadata were previously lost entirely -- those
files fell through to a link with no searchable content -- which left a large
fraction of the EEG/iEEG datasets in a corpus like OpenNeuro effectively
unsearchable.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import re
import struct

import numpy as np

__all__ = [
    "hdf5_digest",
    "vest_header",
    "is_matfile",
    "snirf_digest",
    "edfheader",
    "bvheader",
    "eeglab_digest",
    "mat_digest",
]

_DTYPE_MAP = {
    "float64": "double",
    "float32": "single",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "uint16": "uint16",
    "int32": "int32",
    "uint32": "uint32",
    "int64": "int64",
    "uint64": "uint64",
}


def _jdtype(dtype):
    return _DTYPE_MAP.get(np.dtype(dtype).name, np.dtype(dtype).name)


def _decode(val):
    """Normalise an HDF5 scalar/array value into JSON-friendly Python."""
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, np.ndarray):
        if val.dtype.kind == "S":
            flat = [v.decode("utf-8", "replace") for v in val.ravel().tolist()]
            return flat[0] if val.size == 1 else flat
        if val.dtype.kind == "O":
            return [_decode(v) for v in val.ravel().tolist()]
        out = val.ravel().tolist() if val.size != 1 else val.ravel()[0].item()
        return out
    if isinstance(val, np.generic):
        return val.item()
    return val


# =============================================================================
# HDF5 (SNIRF, NWB, EEGLAB v7.3, generic .h5)
# =============================================================================


def hdf5_digest(filename, maxelem=256, maxdepth=32, exclude=None):
    """Summarise an HDF5 file without reading its bulk arrays.

    Datasets with at most ``maxelem`` elements are inlined verbatim.  Larger
    ones -- and any whose name matches ``exclude`` regardless of size -- are
    replaced by their JData array annotation (``_ArrayType_`` / ``_ArraySize_``)
    with no ``_ArrayData_``, which records the shape and type for search while
    leaving the payload in the content store.

    ``exclude`` matters because a size threshold alone is the wrong test: a
    short recording's signal array may well fit under ``maxelem``, yet a few
    thousand sample values are not metadata and inlining them bloats the
    document without making anything more searchable.
    """
    import h5py

    pattern = re.compile(exclude) if exclude else None

    def visit(node, depth):
        if depth > maxdepth:
            return {}
        out = {}
        attrs = {k: _decode(v) for k, v in node.attrs.items()}
        if attrs:
            out["_Attributes_"] = attrs
        for name, child in node.items():
            if isinstance(child, h5py.Group):
                out[name] = visit(child, depth + 1)
            elif isinstance(child, h5py.Dataset):
                limit = 0 if (pattern and pattern.search(name)) else maxelem
                out[name] = _dataset_digest(child, limit)
        return out

    with h5py.File(filename, "r") as fid:
        return visit(fid, 0)


def _dataset_digest(dset, maxelem):
    shape = list(dset.shape) if dset.shape is not None else []
    numel = int(np.prod(shape)) if shape else 1
    if numel <= maxelem:
        try:
            return _decode(dset[()])
        except Exception:
            pass
    return {
        "_ArrayType_": _jdtype(dset.dtype) if dset.dtype.kind != "S" else "char",
        "_ArraySize_": shape or [1],
        "_ArrayIsTruncated_": True,
    }


#: SNIRF datasets that carry sample data rather than metadata.  Per the SNIRF
#: specification these are ``/nirs(i)/data(j)/{dataTimeSeries,time}`` and the
#: same pair under ``/nirs(i)/aux(k)/``.
SNIRF_BULK = r"^(dataTimeSeries|time|dataOffset)$"


def snirf_digest(filename, maxelem=256):
    """SNIRF metadata digest.

    SNIRF is HDF5, and its searchable content -- probe geometry, wavelengths,
    landmark labels, stimulus timing, the measurement list, and the
    ``metaDataTags`` block -- is all small.  The sample data lives in a handful
    of well-known datasets, which are dropped by name so that a short recording
    does not slip its signal into the document just by being under the element
    threshold.
    """
    digest = hdf5_digest(filename, maxelem=maxelem, exclude=SNIRF_BULK)
    return {"SNIRFData": digest}


def eeglab_digest(filename, maxelem=256):
    """EEGLAB ``.set`` digest, for both HDF5 (v7.3) and MAT v5 containers."""
    with open(filename, "rb") as fid:
        magic = fid.read(8)
    if magic[:8] == b"\x89HDF\r\n\x1a\n":
        return {"EEGLABData": hdf5_digest(filename, maxelem=maxelem)}
    return {"EEGLABData": mat_digest(filename, maxelem=maxelem)}


def is_matfile(filename):
    """True if the file really is a MATLAB container (MAT v5 or v7.3/HDF5).

    A ``.mat`` extension is not a guarantee: FSL and TBSS write plain-text VEST
    design matrices under the same name, and handing one to ``scipy.io`` raises
    "Unknown mat file type".
    """
    with open(filename, "rb") as fid:
        head = fid.read(128)
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return True
    # MAT v5: 116-byte text descriptor, then a 2-byte version and the "MI"/"IM"
    # endian indicator at offset 126
    return len(head) >= 128 and head[126:128] in (b"IM", b"MI")


def vest_header(filename, maxlines=64):
    """Parse the header of an FSL/VEST text matrix (``.mat``, ``.con``, ``.grp``).

    The format is a short run of ``/Key value`` lines terminated by ``/Matrix``,
    after which the numbers begin.  Only the header is read, so the matrix
    itself -- which can be large -- is never loaded.
    """
    header = {}
    rows = 0
    with open(filename, "r", encoding="utf-8", errors="replace") as fid:
        in_matrix = False
        for index, line in enumerate(fid):
            text = line.strip()
            if not in_matrix:
                if index > maxlines and not header:
                    return None  # not a VEST file
                if text == "/Matrix":
                    in_matrix = True
                    continue
                if text.startswith("/"):
                    key, _sep, value = text[1:].partition(" ")
                    header[key] = _vest_value(value.strip())
                continue
            if text:
                rows += 1
    if not header and not rows:
        return None
    header["NumRows"] = rows
    return {"VESTHeader": header}


def _vest_value(value):
    if not value:
        return None
    parts = value.split()
    out = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError:
            try:
                out.append(float(part))
            except ValueError:
                out.append(part)
    return out[0] if len(out) == 1 else out


def mat_digest(filename, maxelem=256):
    """MATLAB v5 ``.mat`` digest: variable names, classes and dimensions.

    Uses ``scipy.io.whosmat`` so the variable payloads are never read.
    """
    try:
        from scipy.io import whosmat
    except ImportError:
        return {"_Unreadable_": "scipy is required to read MAT files"}
    out = {}
    for name, shape, cls in whosmat(filename):
        out[name] = {"_ArrayType_": cls, "_ArraySize_": list(shape)}
    return out


# =============================================================================
# EDF / EDF+ / BDF
# =============================================================================


def edfheader(filename):
    """Parse an EDF, EDF+ or BDF header.

    The layout is fixed-width ASCII: a 256-byte file header followed by ten
    per-signal fields.  Only ``256 + 256 * nsignals`` bytes are read.
    """
    with open(filename, "rb") as fid:
        head = fid.read(256)
        if len(head) < 256:
            raise ValueError("truncated header in %s" % filename)

        is_bdf = head[0:1] == b"\xff"

        def txt(start, length):
            return head[start : start + length].decode("latin-1").strip()

        version = txt(0, 8) if not is_bdf else head[1:8].decode("latin-1").strip()
        nsignals = int(txt(252, 4))
        nrecords = int(txt(236, 8))
        recdur = float(txt(244, 8))

        block = fid.read(256 * nsignals)

    def field(offset, width, count=nsignals):
        base = offset * nsignals
        return [
            block[base + i * width : base + (i + 1) * width].decode("latin-1").strip()
            for i in range(count)
        ]

    labels = field(0, 16)
    transducer = field(16, 80)
    physdim = field(96, 8)
    physmin = field(104, 8)
    physmax = field(112, 8)
    digmin = field(120, 8)
    digmax = field(128, 8)
    prefilter = field(136, 80)
    nsamp = field(216, 8)

    def nums(values, cast=float):
        out = []
        for val in values:
            try:
                out.append(cast(val))
            except ValueError:
                out.append(None)
        return out

    samples = nums(nsamp, int)
    duration = (nrecords * recdur) if nrecords > 0 else None
    srate = [(s / recdur) if (s is not None and recdur) else None for s in samples]

    return {
        "EDFHeader": {
            "Format": "BDF" if is_bdf else "EDF",
            "Version": version,
            "PatientID": txt(8, 80),
            "RecordingID": txt(88, 80),
            "StartDate": txt(168, 8),
            "StartTime": txt(176, 8),
            "HeaderBytes": int(txt(184, 8)),
            "Reserved": txt(192, 44),
            "NumberOfDataRecords": nrecords,
            "DurationOfDataRecord": recdur,
            "NumberOfSignals": nsignals,
            "RecordingDuration": duration,
            "SamplingFrequency": srate,
            "Labels": labels,
            "TransducerType": [t for t in transducer],
            "PhysicalDimension": physdim,
            "PhysicalMinimum": nums(physmin),
            "PhysicalMaximum": nums(physmax),
            "DigitalMinimum": nums(digmin, int),
            "DigitalMaximum": nums(digmax, int),
            "PreFiltering": prefilter,
            "SamplesPerRecord": samples,
        }
    }


# =============================================================================
# BrainVision
# =============================================================================

_BV_SECTION = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")


def bvheader(filename):
    """Parse a BrainVision ``.vhdr`` or ``.vmrk`` sidecar.

    Both are INI-like text files.  ``[Channel Infos]`` and ``[Marker Infos]``
    use comma-separated positional values, which are expanded into named fields
    so that channel labels, units and marker types become searchable.
    """
    sections = {}
    current = None
    with open(filename, "r", encoding="utf-8", errors="replace") as fid:
        for raw in fid:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            match = _BV_SECTION.match(line.strip())
            if match:
                current = match.group("name")
                sections.setdefault(current, {})
                continue
            if current is None:
                continue
            if "=" in line:
                key, _sep, val = line.partition("=")
                sections[current][key.strip()] = val.strip()
            else:
                sections[current].setdefault("_Text_", []).append(line)

    out = {}
    for name, body in sections.items():
        if name == "Channel Infos":
            out["ChannelInfos"] = _bv_channels(body)
        elif name == "Marker Infos":
            out["MarkerInfos"] = _bv_markers(body)
        elif name == "Comment":
            out["Comment"] = "\n".join(body.get("_Text_", []))
        else:
            out[name.replace(" ", "")] = _bv_numeric(body)

    key = "VMRKHeader" if filename.lower().endswith(".vmrk") else "VHDRHeader"
    return {key: out}


def _bv_numeric(body):
    out = {}
    for key, val in body.items():
        if key == "_Text_":
            out["_Text_"] = val
            continue
        try:
            out[key] = int(val)
            continue
        except (TypeError, ValueError):
            pass
        try:
            out[key] = float(val)
            continue
        except (TypeError, ValueError):
            pass
        out[key] = val
    return out


def _bv_channels(body):
    """``Ch1=name,refname,resolution,unit`` -> column-oriented arrays."""
    names, refs, resolutions, units = [], [], [], []
    for key in sorted(body, key=_ch_order):
        if not key.lower().startswith("ch"):
            continue
        parts = _split_bv(body[key])
        names.append(parts[0] if len(parts) > 0 else None)
        refs.append(parts[1] if len(parts) > 1 else None)
        try:
            resolutions.append(float(parts[2]) if len(parts) > 2 and parts[2] else None)
        except ValueError:
            resolutions.append(None)
        units.append(parts[3] if len(parts) > 3 else None)
    return {
        "name": names,
        "reference": refs,
        "resolution": resolutions,
        "unit": units,
    }


def _bv_markers(body):
    """``Mk1=type,description,position,points,channel[,date]`` -> arrays."""
    types, descs, positions, points, channels = [], [], [], [], []
    for key in sorted(body, key=_ch_order):
        if not key.lower().startswith("mk"):
            continue
        parts = _split_bv(body[key])

        def num(idx, cast=int):
            try:
                return cast(parts[idx])
            except (IndexError, ValueError):
                return None

        types.append(parts[0] if parts else None)
        descs.append(parts[1] if len(parts) > 1 else None)
        positions.append(num(2))
        points.append(num(3))
        channels.append(num(4))
    return {
        "type": types,
        "description": descs,
        "position": positions,
        "points": points,
        "channel": channels,
    }


def _split_bv(value):
    """Split a BrainVision comma list, honouring the ``\\1`` escape for commas."""
    return [p.replace("\\1", ",") for p in value.split(",")]


def _ch_order(key):
    match = re.match(r"^[A-Za-z]+(\d+)$", key)
    return (0, int(match.group(1))) if match else (1, key)
