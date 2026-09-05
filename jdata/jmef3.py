"""JMEF3: a JData wrapper for MEF3 (Multiscale Electrophysiology Format) data.

MEF3 stores one recording as a *directory tree* -- session ``.mefd`` holding
channel ``.timd`` directories, each holding segment ``.segd`` directories, each
holding three files: ``.tmet`` metadata, ``.tidx`` index, ``.tdat`` samples.
The wrapper mirrors that tree, using meflib's own level names:

    {
      "MEF3Data": {
        "session_name": "mef3",
        "time_series_channels": {
          "LAD1": {"segments": [{"metadata": {...},
                                 "time_series_indices": ndarray,
                                 "time_series_data":    ndarray}]}
        }
      },
      "MEF3Source": {...}
    }

Field names come from meflib's struct declarations -- MEF3 is fixed-layout
binary with no names in the file (grep a ``.tmet``: zero occurrences of
"sampling_frequency") -- exactly as JNIfTI takes ``dim`` and ``pixdim`` from
``nifti1.h``. Everything is little-endian, unlike CTF.

**Samples are kept as their original RED stream.** RED (Range Encoded
Difference) is MEF's own compressor: first-difference the samples, then range
code the residuals in independently decodable blocks. Measured on a real
89-channel container it reaches 2.31x, where zlib on comparable intracranial
int16 manages 1.23-1.53x and delta+zlib 1.38-1.83x -- so decoding to integers
and re-deflating would *grow* the payload by roughly 1.3-1.5x. Decoding also
requires a RED implementation, which this module does not have. The stream is
therefore stored verbatim, which round-trips exactly; ``PayloadMode`` records
which form is present so a future decoder can offer the other.

**Encryption.** MEF3 supports AES-128 with two access levels. When a password
validation field is set the samples cannot be decoded at all, so ``Encrypted``
is reported without needing to touch the payload.

author: Qianqian Fang <q.fang at neu.edu>
"""

import hashlib
import os
import struct

import numpy as np

__all__ = ["mef32jmef3", "jmef32mef3", "mef3info"]

_UH_BYTES = 1024
_INDEX_RECORD = 56  # verified: file_offset[last] + block_bytes[last] == .tdat size

# Universal header, verified by reading real files.
_UH = [
    ("header_CRC", 0, "<I", 4),
    ("body_CRC", 4, "<I", 4),
    ("mef_version_major", 13, "B", 1),
    ("mef_version_minor", 14, "B", 1),
    ("byte_order_code", 15, "B", 1),
    ("start_time", 16, "<q", 8),
    ("end_time", 24, "<q", 8),
    ("number_of_entries", 32, "<q", 8),
    ("maximum_entry_size", 40, "<q", 8),
    ("segment_number", 48, "<i", 4),
]
_UH_TEXT = [
    ("file_type_string", 8, 5),
    ("channel_name", 52, 256),
    ("session_name", 308, 256),
    ("anonymized_name", 564, 256),
]
_UH_UUID = [("level_UUID", 820), ("file_UUID", 836), ("provenance_UUID", 852)]
_PW1, _PW2 = 868, 884

# metadata_section_2 offsets, located by matching against the BIDS sidecar
# (sampling_frequency 2048.0, AC_line_frequency 60.0 == PowerLineFrequency).
_S2 = [
    ("sampling_frequency", 8720),
    ("low_frequency_filter_setting", 8728),
    ("high_frequency_filter_setting", 8736),
    ("notch_filter_frequency_setting", 8744),
    ("AC_line_frequency", 8752),
]


def _text(blob):
    return blob.split(b"\0")[0].decode("latin1", "replace").strip()


def _universal_header(raw):
    """Parse the 1024-byte universal header that starts every MEF3 file."""
    if len(raw) < _UH_BYTES:
        raise ValueError("short MEF3 universal header (%d bytes)" % len(raw))
    out = {}
    for name, off, fmt, size in _UH:
        out[name] = struct.unpack(fmt, raw[off : off + size])[0]
    for name, off, size in _UH_TEXT:
        out[name] = _text(raw[off : off + size])
    for name, off in _UH_UUID:
        out[name] = raw[off : off + 16].hex()
    out["level_1_encrypted"] = any(raw[_PW1 : _PW1 + 16])
    out["level_2_encrypted"] = any(raw[_PW2 : _PW2 + 16])
    return out


def _section2(raw):
    out = {}
    for name, off in _S2:
        if len(raw) >= off + 8:
            val = struct.unpack("<d", raw[off : off + 8])[0]
            # meflib writes -1.0 for "not entered"
            if val == val:
                out[name] = val
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(mefd):
    """Yield (channel, segment_dir, {ext: relpath}) for every segment."""
    for chan in sorted(os.listdir(mefd)):
        cdir = os.path.join(mefd, chan)
        if not os.path.isdir(cdir) or not chan.endswith((".timd", ".vidd")):
            continue
        name = chan.rsplit(".", 1)[0]
        for seg in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, seg)
            if not os.path.isdir(sdir) or not seg.endswith(".segd"):
                continue
            files = {}
            for f in sorted(os.listdir(sdir)):
                if "." in f:
                    files[f.rsplit(".", 1)[-1]] = os.path.join(chan, seg, f)
            yield name, seg, files


def mef3info(mefd):
    """The small summary that sits beside ``_DataLink_`` in a document."""
    mefd = mefd.rstrip(os.sep)
    nchan = 0
    freq = None
    encrypted = False
    version = None
    for _, _, files in _walk(mefd):
        nchan += 1
        if "tmet" not in files:
            continue
        with open(os.path.join(mefd, files["tmet"]), "rb") as fid:
            raw = fid.read(_UH_BYTES + 16384)
        uh = _universal_header(raw)
        encrypted = encrypted or uh["level_1_encrypted"] or uh["level_2_encrypted"]
        version = version or "%d.%d" % (uh["mef_version_major"], uh["mef_version_minor"])
        if freq is None:
            freq = _section2(raw).get("sampling_frequency")
    return {
        "Format": "MEF3",
        "MEFVersion": version,
        "Encrypted": encrypted,
        "Compression": "RED",
    }


def mef32jmef3(mefd, **kwargs):
    """Read a MEF3 ``.mefd`` directory into a JMEF3 structure."""
    mefd = mefd.rstrip(os.sep)
    if not os.path.isdir(mefd):
        raise ValueError("%s is not a MEF3 .mefd directory" % mefd)

    channels = {}
    files = []
    session = None
    encrypted = False

    for name, segname, members in _walk(mefd):
        node = {"segment": segname}
        for ext in ("tmet", "tidx", "tdat", "vmet", "vidx"):
            rel = members.get(ext)
            if rel is None:
                continue
            full = os.path.join(mefd, rel)
            with open(full, "rb") as fid:
                blob = fid.read()
            files.append(
                {"Name": rel, "Bytes": len(blob), "SHA256": hashlib.sha256(blob).hexdigest()}
            )
            if ext in ("tmet", "vmet"):
                uh = _universal_header(blob)
                session = session or uh["session_name"]
                encrypted = encrypted or uh["level_1_encrypted"] or uh["level_2_encrypted"]
                node["metadata"] = {
                    "universal_header": uh,
                    "section_2": _section2(blob),
                    "_ByteStream_": blob,
                }
            elif ext in ("tidx", "vidx"):
                body = blob[_UH_BYTES:]
                nrec = len(body) // _INDEX_RECORD
                node["time_series_indices"] = {
                    "universal_header": _universal_header(blob),
                    "_UniversalHeaderBytes_": blob[:_UH_BYTES],
                    "records": np.frombuffer(
                        body[: nrec * _INDEX_RECORD], dtype=np.uint8
                    ).reshape(nrec, _INDEX_RECORD),
                }
            else:
                node["time_series_data"] = {
                    "universal_header": _universal_header(blob),
                    "_UniversalHeaderBytes_": blob[:_UH_BYTES],
                    "RED": np.frombuffer(blob[_UH_BYTES:], dtype=np.uint8),
                }
        channels.setdefault(name, {"segments": []})["segments"].append(node)

    # session-level records (.rdat/.ridx) and anything else living in the root
    extra = {}
    for f in sorted(os.listdir(mefd)):
        full = os.path.join(mefd, f)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as fid:
            blob = fid.read()
        files.append({"Name": f, "Bytes": len(blob), "SHA256": hashlib.sha256(blob).hexdigest()})
        extra[f] = {"_ByteStream_": blob}

    data = {"session_name": session, "time_series_channels": channels}
    if extra:
        data["records"] = extra

    return {
        "MEF3Data": data,
        "MEF3Source": {
            "Format": "MEF3",
            "Container": os.path.basename(mefd),
            "ContainerType": "directory",
            "Endian": "little",
            "PayloadMode": "verbatim",
            "Compression": "RED",
            "Encrypted": encrypted,
            "IndexRecordBytes": _INDEX_RECORD,
            "Files": files,
        },
    }


def jmef32mef3(jmef3, mefd):
    """Rebuild the ``.mefd`` tree. Returns {relative path: sha256}."""
    data = jmef3["MEF3Data"]
    src = jmef3.get("MEF3Source", {})
    byname = {f["Name"]: f for f in src.get("Files", [])}
    written = {}

    def put(rel, blob):
        full = os.path.join(mefd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fid:
            fid.write(blob)
        written[rel] = hashlib.sha256(blob).hexdigest()

    os.makedirs(mefd, exist_ok=True)

    for chan, node in data.get("time_series_channels", {}).items():
        for seg in node["segments"]:
            segname = seg["segment"]
            base = "%s.timd/%s" % (chan, segname)
            stem = segname.rsplit(".", 1)[0]
            if "metadata" in seg:
                put("%s/%s.tmet" % (base, stem), bytes(seg["metadata"]["_ByteStream_"]))
            if "time_series_indices" in seg:
                idx = seg["time_series_indices"]
                blob = bytes(idx["_UniversalHeaderBytes_"]) + np.asarray(
                    idx["records"], dtype=np.uint8
                ).tobytes()
                put("%s/%s.tidx" % (base, stem), blob)
            if "time_series_data" in seg:
                dat = seg["time_series_data"]
                blob = bytes(dat["_UniversalHeaderBytes_"]) + np.asarray(
                    dat["RED"], dtype=np.uint8
                ).tobytes()
                put("%s/%s.tdat" % (base, stem), blob)

    for name, node in data.get("records", {}).items():
        put(name, bytes(node["_ByteStream_"]))

    return written

