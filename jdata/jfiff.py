"""JFIFF: a JData wrapper for Neuromag/Elekta FIFF (``.fif``) recordings.

A FIFF file is a flat stream of tags. Each tag is a 16-byte big-endian header

    kind (i4)  type (i4)  size (i4)  next (i4)

followed by ``size`` payload bytes. ``next`` links the tags into a tree: 0
means "the next tag follows immediately", a positive value is an explicit file
offset, and a negative value ends a block.

    {
      "FIFFData": {
        "TagDirectory": ndarray [ntag, 4],   # kind, type, size, next -- verbatim
        "Tags":    [ {...}, ... ],           # small tags, payload kept as bytes
        "TagData": ndarray                   # every bulk payload, concatenated
      },
      "FIFFSource": {...}
    }

**Why this shape rather than a named tree.** FIFF has no open vendor
specification. The names everyone uses (``FIFF_SFREQ``, ``FIFFB_MEAS_INFO``)
come from Neuromag's proprietary ``fiff_file.h`` by way of MNE-C and
MNE-Python's ``constants.py`` -- a community reimplementation, not a published
standard, unlike ``nifti1.h`` for NIfTI or ``meflib.h`` for MEF3. This module
therefore treats the **numeric kind as ground truth** and any symbolic name as
advisory decoration: a round-trip depends only on the numbers, so a wrong or
missing name can never corrupt the data.

For the same reason the bulk payloads are lifted by **size**, not by kind. On a
real recording, tags larger than the threshold were 1800 of 2026 tags and
99.99% of the bytes, and identifying them needs no constant table at all.

Payload bytes are kept as ``uint8``. Decoding them to their declared element
type would require trusting the ``FIFFT_*`` type codes, which are from the same
unverified table; the raw bytes plus the recorded ``type`` are lossless and let
a later, verified decoder do it properly.

author: Qianqian Fang <q.fang at neu.edu>
"""

import hashlib
import os
import struct

import numpy as np

__all__ = ["fiff2jfiff", "jfiff2fiff", "fiffinfo", "FIFF_KIND_NAMES"]

_TAGHDR = 16
_BULK_BYTES = 65536  # lift anything at least this large into its own array

# Advisory only. Sourced from MNE-Python's public constants table; FIFF has no
# vendor specification to check these against, so they annotate and never
# drive behaviour. Corroborated on a real file: kind 203 appeared 99 times in a
# 99-channel recording, 213 appeared 102 times, 300 held 99.99% of the bytes.
FIFF_KIND_NAMES = {
    100: "FIFF_FILE_ID",
    101: "FIFF_DIR_POINTER",
    103: "FIFF_BLOCK_ID",
    104: "FIFF_BLOCK_START",
    105: "FIFF_BLOCK_END",
    106: "FIFF_FREE_LIST",
    200: "FIFF_NCHAN",
    201: "FIFF_SFREQ",
    202: "FIFF_DATA_PACK",
    203: "FIFF_CH_INFO",
    204: "FIFF_MEAS_DATE",
    208: "FIFF_FIRST_SAMPLE",
    209: "FIFF_LAST_SAMPLE",
    213: "FIFF_DIG_POINT",
    219: "FIFF_LOWPASS",
    222: "FIFF_COORD_TRANS",
    223: "FIFF_HIGHPASS",
    300: "FIFF_DATA_BUFFER",
    301: "FIFF_DATA_SKIP",
}


def _walk_tags(raw):
    """Yield (offset, kind, type, size, next) for every tag in the stream."""
    total = len(raw)
    off = 0
    seen = set()
    while off + _TAGHDR <= total:
        if off in seen:  # a malformed 'next' chain must not loop forever
            break
        seen.add(off)
        kind, typ, size, nxt = struct.unpack(">4i", raw[off : off + _TAGHDR])
        if size < 0 or off + _TAGHDR + size > total:
            break
        yield off, kind, typ, size, nxt
        off = nxt if nxt > 0 else off + _TAGHDR + size


def fiffinfo(filename):
    """The small summary that sits beside ``_DataLink_`` in a document."""
    with open(filename, "rb") as fid:
        raw = fid.read()
    ntag = nbulk = bulkbytes = 0
    kinds = set()
    for _, kind, _, size, _ in _walk_tags(raw):
        ntag += 1
        kinds.add(kind)
        if size >= _BULK_BYTES:
            nbulk += 1
            bulkbytes += size
    return {
        "Format": "FIFF",
        "Endian": "big",
        "NumberOfTags": ntag,
        "NumberOfBulkTags": nbulk,
        "BulkBytes": bulkbytes,
        "DistinctKinds": len(kinds),
    }


def fiff2jfiff(filename, **kwargs):
    """Read a ``.fif`` file into a JFIFF structure."""
    with open(filename, "rb") as fid:
        raw = fid.read()

    directory = []
    tags = []
    chunks = []
    filled = 0
    for off, kind, typ, size, nxt in _walk_tags(raw):
        directory.append([kind, typ, size, nxt])
        payload = raw[off + _TAGHDR : off + _TAGHDR + size]
        entry = {"Index": len(directory) - 1, "Kind": kind, "Type": typ}
        name = FIFF_KIND_NAMES.get(kind)
        if name:
            entry["KindName"] = name
        if size >= _BULK_BYTES:
            # One concatenated array, not one array per tag. A raw recording
            # holds ~1800 data buffers of ~200 KB; compressed separately every
            # one of them falls below the block size, so none gets an
            # _ArrayZipOffsets_ index and the whole payload loses parallel
            # inflate. Concatenated, the stream is indexed and blocks span it.
            entry["DataOffset"] = filled
            chunks.append(payload)
            filled += size
        else:
            entry["_ByteStream_"] = payload
        tags.append(entry)

    consumed = sum(_TAGHDR + d[2] for d in directory)
    trailer = raw[consumed:] if consumed < len(raw) else b""

    data = {
        "TagDirectory": np.asarray(directory, dtype=np.int32).reshape(-1, 4),
        "Tags": tags,
    }
    if chunks:
        data["TagData"] = np.frombuffer(b"".join(chunks), dtype=np.uint8)

    source = {
        "Format": "FIFF",
        "Endian": "big",
        "ContainerType": "file",
        "File": os.path.basename(filename),
        "Bytes": len(raw),
        "SHA256": hashlib.sha256(raw).hexdigest(),
        "BulkThresholdBytes": _BULK_BYTES,
        "ConstantsSource": "mne-python (advisory names only)",
    }
    if trailer:
        source["Trailer"] = {"_ByteStream_": trailer}
    return {"FIFFData": data, "FIFFSource": source}


def jfiff2fiff(jfiff, filename):
    """Rebuild the ``.fif`` file. Returns the SHA-256 of the result."""
    data = jfiff["FIFFData"]
    src = jfiff.get("FIFFSource", {})
    directory = np.asarray(data["TagDirectory"], dtype=np.int32).reshape(-1, 4)
    bulk = np.asarray(data.get("TagData", []), dtype=np.uint8)

    out = bytearray()
    for row, tag in zip(directory, data["Tags"]):
        kind, typ, size, nxt = (int(v) for v in row)
        out += struct.pack(">4i", kind, typ, size, nxt)
        if "DataOffset" in tag:
            start = int(tag["DataOffset"])
            out += bulk[start : start + size].tobytes()
        else:
            out += bytes(tag["_ByteStream_"])

    trailer = src.get("Trailer")
    if trailer:
        out += bytes(trailer["_ByteStream_"])

    with open(filename, "wb") as fid:
        fid.write(out)
    return hashlib.sha256(bytes(out)).hexdigest()
