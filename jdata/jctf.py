"""JCTF: a JData wrapper for CTF MEG ``.ds`` recordings.

A CTF recording is a *directory*, not a file, so the wrapper mirrors the
directory: every file in the ``.ds`` becomes a key, exactly as an HDF5 group
name becomes a key in JSNIRF. Nothing below the container key is renamed.

    {
      "CTFData": {
        "res4": {...},          # parsed acquisition header
        "meg4": ndarray,        # the samples, [trial, channel, sample]
        "hc": {...}, "acq": {...}, "MarkerFile.mrk": {...}, ...
      },
      "CTFSource": {...}        # what a byte-exact rebuild needs
    }

Field names come from three different places, and which one applies depends on
the file:

* ``.infods``, ``.acq``, ``MarkerFile.mrk``, ``params.dsc`` are tagged or
  key/value ASCII -- the names are *in* the file, so they are read, never
  invented.
* ``res4`` is a fixed-layout binary struct with no names in it at all (grep it:
  there are zero occurrences of "no_samples" or "sample_rate"). Its names come
  from CTF's published struct declaration, the same way JNIfTI takes ``dim``
  and ``pixdim`` from ``nifti1.h``.
* ``.hc`` names its contents in English prose ("standard nasion coil position
  relative to dewar (cm):"), which is not a key set. It is kept as bytes.

Everything is big-endian, unlike MEF3.

**Losslessness.** Encoding does not keep the original bytes in the content
store, so this wrapper is the only remaining description of the recording.
Every non-bulk file is therefore kept verbatim as a ``_ByteStream_`` beside its
parsed view, which costs 0.13% of a container (a 3.19 MB ``res4`` against a
2.42 GB ``meg4``) and makes a byte-exact rebuild a property of the container
rather than of this parser being complete.

author: Qianqian Fang <q.fang at neu.edu>
"""

import hashlib
import os
import re
import struct

import numpy as np

__all__ = ["ctf2jctf", "jctf2ctf", "ctfinfo"]

# res4 offsets, verified against 7 datasets. Two invariants pin them:
#   epoch_time == no_trials * no_samples / sample_rate
#   meg4 data bytes == no_trials * no_channels * no_samples * 4
_RES4_MAGIC = b"MEG42RS\x00"
_MEG4_MAGIC = b"MEG41CP\x00"
_OFF_APPNAME = 8
_OFF_ORIGIN = 264
_OFF_DESC = 520
_OFF_TRIALS_AVGD = 776
_OFF_TIME = 778
_OFF_DATE = 1033
_OFF_NSAMP = 1288
_OFF_NCHAN = 1292
_OFF_SRATE = 1296
_OFF_EPOCH = 1304
_OFF_NTRIALS = 1312
# Channel names sit at a fixed offset in every file examined, 32 bytes each,
# NUL padded, immediately followed by the per-sensor records.
_OFF_CHANNAMES = 1847
_NAMELEN = 32

_NAME_RE = re.compile(rb"^[A-Za-z][A-Za-z0-9_\-\+\.]{0,31}$")


def _text(blob):
    """A NUL-padded fixed-width field as text."""
    return blob.split(b"\0")[0].decode("latin1", "replace").strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meg4_parts(dspath):
    """``.meg4`` plus its 2 GB continuation files, in acquisition order.

    CTF splits the sample stream at 2 GB into ``.1_meg4``, ``.2_meg4`` and so
    on; the parts are one logical array and must be concatenated in order.
    """
    parts = []
    for name in os.listdir(dspath):
        if name.endswith(".meg4"):
            parts.append((0, name))
        else:
            hit = re.match(r"^.*\.(\d+)_meg4$", name)
            if hit:
                parts.append((int(hit.group(1)), name))
    return [name for _, name in sorted(parts)]


def _parse_res4(raw):
    """The fields whose offsets are verified; the rest stays in _ByteStream_."""
    if raw[:8] != _RES4_MAGIC:
        raise ValueError("not a CTF res4 file (bad magic %r)" % raw[:8])

    head = {
        "appName": _text(raw[_OFF_APPNAME : _OFF_APPNAME + 256]),
        "dataOrigin": _text(raw[_OFF_ORIGIN : _OFF_ORIGIN + 256]),
        "dataDescription": _text(raw[_OFF_DESC : _OFF_DESC + 256]),
        "no_trials_avgd": struct.unpack(">h", raw[_OFF_TRIALS_AVGD : _OFF_TRIALS_AVGD + 2])[0],
        "data_time": _text(raw[_OFF_TIME : _OFF_TIME + 255]),
        "data_date": _text(raw[_OFF_DATE : _OFF_DATE + 255]),
        "no_samples": struct.unpack(">i", raw[_OFF_NSAMP : _OFF_NSAMP + 4])[0],
        "no_channels": struct.unpack(">h", raw[_OFF_NCHAN : _OFF_NCHAN + 2])[0],
        "sample_rate": struct.unpack(">d", raw[_OFF_SRATE : _OFF_SRATE + 8])[0],
        "epoch_time": struct.unpack(">d", raw[_OFF_EPOCH : _OFF_EPOCH + 8])[0],
        "no_trials": struct.unpack(">h", raw[_OFF_NTRIALS : _OFF_NTRIALS + 2])[0],
    }

    nchan = head["no_channels"]
    names = []
    for k in range(max(0, nchan)):
        start = _OFF_CHANNAMES + _NAMELEN * k
        field = raw[start : start + _NAMELEN]
        if len(field) < _NAMELEN:
            break
        names.append(_text(field))
    if names:
        head["chanName"] = names

    return head


def _parse_markerfile(text):
    """MarkerFile.mrk names its own fields, so they are read rather than coined."""
    out = {"markers": []}
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    cur = None
    idx = 0
    while idx < len(lines):
        key = lines[idx].strip()
        if key.endswith(":") and idx + 1 < len(lines):
            name = key[:-1].strip()
            value = lines[idx + 1].strip()
            if name == "CLASSGROUPID":
                cur = {}
                out["markers"].append(cur)
            if name == "LIST OF SAMPLES":
                samples = []
                idx += 2  # step past "LIST OF SAMPLES:" and the column titles
                while idx < len(lines):
                    row = lines[idx].split()
                    if len(row) != 2:
                        break
                    try:
                        samples.append([int(row[0]), float(row[1])])
                    except ValueError:
                        break
                    idx += 1
                if cur is not None:
                    cur["samples"] = samples
                continue
            target = cur if cur is not None else out
            try:
                target[name] = int(value)
            except ValueError:
                target[name] = value
            idx += 2
            continue
        idx += 1
    return out


def _read_samples(dspath, parts, nchan, nsamp, ntrials, maxsamples=None):
    """Concatenate the ``.meg4`` parts into [trial, channel, sample] int32.

    Verified layout: channel-major within a trial. Autocorrelation alone gets
    this wrong on empty-room noise, where there is no signal to correlate; the
    reliable evidence is that under this layout unused channels have exactly
    zero variance while the others do not.
    """
    blobs = []
    for name in parts:
        with open(os.path.join(dspath, name), "rb") as fid:
            magic = fid.read(8)
            if magic != _MEG4_MAGIC:
                raise ValueError("bad meg4 magic %r in %s" % (magic, name))
            blobs.append(fid.read())

    data = b"".join(blobs)
    per = nchan * nsamp
    if per <= 0:
        raise ValueError("res4 declares %d channels and %d samples" % (nchan, nsamp))

    available = len(data) // (per * 4)
    if ntrials <= 0 or ntrials > available:
        ntrials = available
    if maxsamples:
        ntrials = min(ntrials, max(1, int(maxsamples) // per))

    want = ntrials * per * 4
    arr = np.frombuffer(data[:want], dtype=">i4").astype(np.int32)
    return arr.reshape(ntrials, nchan, nsamp), [len(b) for b in blobs]


def ctfinfo(dspath):
    """The small summary that sits beside ``_DataLink_`` in a document.

    Only fields that are absent from the BIDS sidecars, non-identifying, and
    that change what a client does *before* fetching the attachment.
    """
    res4 = [f for f in os.listdir(dspath) if f.endswith(".res4")]
    if not res4:
        raise ValueError("no .res4 in %s" % dspath)
    with open(os.path.join(dspath, res4[0]), "rb") as fid:
        head = _parse_res4(fid.read(1 << 16))
    return {
        "Format": "CTF",
        "NumberOfSamples": head["no_samples"],
        "NumberOfTrials": head["no_trials"],
        "SplitParts": len(_meg4_parts(dspath)),
    }


def ctf2jctf(dspath, maxsamples=None, **kwargs):
    """Read a CTF ``.ds`` directory into a JCTF structure."""
    dspath = dspath.rstrip(os.sep)
    if not os.path.isdir(dspath):
        raise ValueError("%s is not a CTF .ds directory" % dspath)

    entries = sorted(os.listdir(dspath))
    res4name = next((f for f in entries if f.endswith(".res4")), None)
    if res4name is None:
        raise ValueError("no .res4 in %s" % dspath)

    with open(os.path.join(dspath, res4name), "rb") as fid:
        res4raw = fid.read()
    head = _parse_res4(res4raw)

    parts = _meg4_parts(dspath)
    if not parts:
        raise ValueError("no .meg4 payload in %s" % dspath)

    samples, partsizes = _read_samples(
        dspath,
        parts,
        head["no_channels"],
        head["no_samples"],
        head["no_trials"],
        maxsamples,
    )

    # res4 is kept whole: only a fraction of its 3.19 MB has verified offsets
    # (the per-sensor records and a ~2.78 MB trailing block are not yet
    # mapped), so the bytes are what guarantees a rebuild, not the parse.
    head["_ByteStream_"] = res4raw
    data = {"res4": head, "meg4": samples}

    files = [{"Name": res4name, "Bytes": len(res4raw), "SHA256": hashlib.sha256(res4raw).hexdigest()}]
    for name in parts:
        full = os.path.join(dspath, name)
        files.append({"Name": name, "Bytes": os.path.getsize(full), "SHA256": _sha256(full), "Role": "data"})

    for name in entries:
        if name == res4name or name in parts:
            continue
        full = os.path.join(dspath, name)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as fid:
            blob = fid.read()
        files.append({"Name": name, "Bytes": len(blob), "SHA256": hashlib.sha256(blob).hexdigest()})

        key = name.split(".")[-1] if name.startswith(os.path.basename(dspath)[:8]) else name
        if name.endswith((".hc", ".acq", ".infods")):
            key = name.rsplit(".", 1)[-1]
        node = {"_ByteStream_": blob}
        if name == "MarkerFile.mrk":
            try:
                node.update(_parse_markerfile(blob.decode("latin1")))
            except Exception:
                pass
        elif name == "BadChannels":
            try:
                node["channels"] = [ln.strip() for ln in blob.decode("latin1").split("\n") if ln.strip()]
            except Exception:
                pass
        data[key] = node

    return {
        "CTFData": data,
        "CTFSource": {
            "Format": "CTF",
            "Container": os.path.basename(dspath),
            "ContainerType": "directory",
            "SampleLayout": "trial-channel-sample",
            "Endian": "big",
            "MEG4Parts": parts,
            "MEG4PartBytes": partsizes,
            "Files": files,
        },
    }


def jctf2ctf(jctf, dspath):
    """Rebuild the ``.ds`` directory. Returns {filename: sha256} of the result."""
    data = jctf["CTFData"]
    src = jctf.get("CTFSource", {})
    os.makedirs(dspath, exist_ok=True)

    byname = {f["Name"]: f for f in src.get("Files", [])}
    written = {}

    res4name = next((n for n in byname if n.endswith(".res4")), None)
    if res4name is None:
        raise ValueError("CTFSource lists no .res4")
    blob = bytes(data["res4"]["_ByteStream_"])
    with open(os.path.join(dspath, res4name), "wb") as fid:
        fid.write(blob)
    written[res4name] = hashlib.sha256(blob).hexdigest()

    parts = src.get("MEG4Parts", [])
    partbytes = src.get("MEG4PartBytes", [])
    payload = np.ascontiguousarray(np.asarray(data["meg4"], dtype=np.int32)).astype(">i4").tobytes()
    pos = 0
    for idx, name in enumerate(parts):
        take = partbytes[idx] if idx < len(partbytes) else len(payload) - pos
        chunk = _MEG4_MAGIC + payload[pos : pos + take]
        pos += take
        with open(os.path.join(dspath, name), "wb") as fid:
            fid.write(chunk)
        written[name] = hashlib.sha256(chunk).hexdigest()

    for key, node in data.items():
        if key in ("res4", "meg4"):
            continue
        if not isinstance(node, dict) or "_ByteStream_" not in node:
            continue
        name = next((n for n in byname if n == key or n.endswith("." + key)), key)
        blob = bytes(node["_ByteStream_"])
        with open(os.path.join(dspath, name), "wb") as fid:
            fid.write(blob)
        written[name] = hashlib.sha256(blob).hexdigest()

    return written
