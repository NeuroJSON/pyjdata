"""
Re-encode modality-specific files into binary JData attachments.

The digest keeps a file's *header* inline and searchable; its bulk array data is
written out as a compressed binary JData document (BJData, per the JData
specification) and referenced by content hash.  This is what makes a NeuroJSON
dataset self-describing all the way down: the payload is no longer an opaque
vendor blob but a JData container any JData reader can open, in the same family
of formats as the digest itself.

============================  ==========  ======================================
source                        attachment  bulk keys moved into the attachment
============================  ==========  ======================================
``.nii`` ``.nii.gz``          ``.bnii``   ``NIFTIData``, ``NIFTIExtension``
``.hdr``/``.img``             ``.bnii``   ``NIFTIData``
``.snirf``                    ``.bnirs``  ``SNIRFData``
``.gii``                      ``.bgii``   ``GIFTIData``
``.jmsh`` ``.msh`` ``.off``   ``.bmsh``   mesh arrays
``.mat`` ``.set``             ``.jdb``    every variable
============================  ==========  ======================================

Attachments are named ``<sha256>_<codec>.<ext>`` where the digest is of the
**original** file's bytes.  Naming by the source rather than by the encoded
output is deliberate: it stays stable if the encoder or its compression settings
ever change, it lets two dataset versions that share a file share one
attachment, and it means the name can be predicted from the source alone.  The
encoded payload's own digest is recorded alongside it in the manifest, so what
was served can still be verified.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import io
import hashlib

__all__ = [
    "encode_attachment",
    "attachment_name",
    "ENCODABLE",
    "encoder_for",
]

#: source extension -> (attachment extension, bulk keys to move out)
# Attachment extensions are the *binary* JData spellings: every payload here is
# serialised with dumpb (BJData), and NeuroJSON names binary forms b* against
# the text j* -- .bnii/.jnii, .bmsh/.jmsh, .bnirs/.jnirs. A j* name on a
# binary payload mislabels the file to every reader that dispatches on suffix.
ENCODABLE = {
    ".nii": (".bnii", ("NIFTIData", "NIFTIExtension")),
    ".nii.gz": (".bnii", ("NIFTIData", "NIFTIExtension")),
    ".hdr": (".bnii", ("NIFTIData", "NIFTIExtension")),
    ".img": (".bnii", ("NIFTIData", "NIFTIExtension")),
    ".img.gz": (".bnii", ("NIFTIData", "NIFTIExtension")),
    ".snirf": (".bnirs", ("SNIRFData",)),
    ".gii": (".bgii", ("GIFTIData",)),
    ".jmsh": (".bmsh", ()),
    ".mat": (".jdb", ()),
    # electrophysiology: header and per-channel calibration stay inline, the
    # sample array is the bulk payload. .set was previously read by the generic
    # MAT loader, which cannot follow the .fdt companion that holds the samples.
    ".set": (".beeg", ("EEGData",)),
    ".edf": (".beeg", ("EEGData",)),
    ".bdf": (".beeg", ("EEGData",)),
    ".vhdr": (".beeg", ("EEGData",)),
    # MEG/iEEG recordings that are directories rather than files. CTF and MEF3
    # name their parts with the filesystem, so the container *is* the tree.
    ".fif": (".bfif", ("FIFFData",)),
    ".ds": (".bmeg", ("CTFData",)),
    ".mefd": (".bmef", ("MEF3Data",)),
}

#: source extensions that name a directory, not a file
CONTAINER_EXT = frozenset((".ds", ".mefd"))


def container_digest(path, algo="sha256"):
    """A stable content hash for a directory-shaped recording.

    ``.ds`` and ``.mefd`` are directories, so the file digest the CAS uses to
    name an attachment does not apply. Hashing the sorted list of member
    ``relpath`` and member digest gives the same properties: deterministic,
    independent of mtime, path and host, and changing if any byte of any member
    changes -- so two conversions of the same recording still agree, which is
    what makes the output DOI-capable.
    """
    import hashlib

    members = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h = hashlib.new(algo)
            with open(full, "rb") as fid:
                for chunk in iter(lambda: fid.read(1 << 20), b""):
                    h.update(chunk)
            members.append((rel, h.hexdigest()))

    top = hashlib.new(algo)
    for rel, digest in members:
        top.update(("%s %s\n" % (rel, digest)).encode("utf-8"))
    return top.hexdigest(), len(members)


def encoder_for(ext):
    """Return ``(attachment_ext, bulk_keys)`` for a source extension, or None."""
    return ENCODABLE.get(ext)


def attachment_name(digest, compression, ext):
    """``<sha256>_<codec><ext>`` -- e.g. ``a1b2..._zlib.bnii``.

    ``compression`` may be empty, giving ``<sha256><ext>``.
    """
    codec = ("_%s" % compression) if compression else ""
    return "%s%s%s" % (digest, codec, ext)


def encode_attachment(path, ext, compression="zlib", nthread=1, **kwargs):
    """Read a modality file and produce its binary JData attachment.

    Returns ``(header, payload, attachment_ext, bulk_keys)`` where

    ``header``
        the parsed structure with the bulk arrays *removed*, to be inlined in
        the document;
    ``payload``
        the serialised, compressed BJData bytes of the whole structure,
        including the bulk arrays, to be stored as the attachment;
    ``bulk_keys``
        the keys that were moved out, so the caller can point a ``_DataLink_``
        at each of them.

    Raises on anything it cannot parse; the caller decides whether to fall back
    to referencing the original file.
    """
    spec = ENCODABLE.get(ext)
    if spec is None:
        raise ValueError("no attachment encoder for %r" % ext)
    attach_ext, bulk_keys = spec

    data = _load(path, ext, **kwargs)
    if not isinstance(data, dict):
        data = {"Data": data}

    payload = _serialise(data, compression, nthread)

    present = tuple(k for k in bulk_keys if k in data)
    if not present:
        # formats without a fixed bulk key (mesh, .mat): everything large is in
        # the attachment, so the header is the key list alone
        header = {}
    else:
        header = {k: v for k, v in data.items() if k not in present}
    return header, payload, attach_ext, (present or tuple(data.keys()))


def _load(path, ext, **kwargs):
    if ext in (".nii", ".nii.gz", ".hdr", ".img", ".img.gz"):
        from .jnifti import nii2jnii

        return nii2jnii(path)
    if ext == ".snirf":
        from .jfile import loadsnirf

        return {"SNIRFData": loadsnirf(path)}
    if ext == ".gii":
        from .jgifti import gii2jgii

        return gii2jgii(path)
    if ext == ".jmsh":
        from .jfile import loadjson

        return loadjson(path)
    if ext in (".edf", ".bdf", ".vhdr", ".set"):
        from .jeeg import eeg2jeeg

        return eeg2jeeg(path)
    if ext == ".fif":
        from .jfiff import fiff2jfiff

        return fiff2jfiff(path)
    if ext == ".ds":
        from .jctf import ctf2jctf

        return ctf2jctf(path)
    if ext == ".mefd":
        from .jmef3 import mef32jmef3

        return mef32jmef3(path)
    if ext == ".mat":
        return _load_mat(path)
    raise ValueError("no loader for %r" % ext)


def _load_mat(path):
    with open(path, "rb") as fid:
        magic = fid.read(8)
    if magic == b"\x89HDF\r\n\x1a\n":
        from .h5 import loadh5

        return loadh5(path)
    from scipy.io import loadmat

    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _serialise(data, compression, nthread=1):
    """JData-encode a structure and serialise it as BJData bytes.

    ``encode`` applies the JData array annotations and the compression codec, so
    every array in the attachment arrives as ``_ArrayZipData_`` rather than a
    raw buffer -- the payload stays a valid JData document, readable by any
    JData implementation, not just an opaque blob with a JSON wrapper.
    """
    from .jdata import encode as jdata_encode
    from .jfile import dumpb

    annotated = jdata_encode(
        data,
        compression=compression or None,
        compressarraysize=0,
        nthread=nthread,
    )
    return bytes(dumpb(annotated))


def payload_digest(payload, algo="sha256"):
    """Digest of the encoded attachment itself, for verification."""
    return hashlib.new(algo, payload).hexdigest()
