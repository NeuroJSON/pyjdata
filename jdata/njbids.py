"""
BIDS dataset to version-invariant JSON digest conversion.

Produces a single JSON document per dataset in which every *searchable* piece of
metadata is inlined in human-readable form, and every *bulky* payload is replaced
by an immutable content-addressed ``_DataLink_`` (see :mod:`jdata.njcas`).

Three properties distinguish this from the older ``njprep`` pipeline:

**Version invariance.**  The document is a pure function of the dataset's git
tree.  Nothing time-, host- or path-dependent enters the payload, keys are
emitted in a fixed order, and attachment identifiers are content hashes rather
than hashes of a file's path.  Re-running the conversion on unchanged input
reproduces the document byte for byte, so the accompanying ``fingerprint`` is a
stable identifier that a DOI can be minted against.

**No re-encoding.**  Bulky files are registered in the store as their original
bytes instead of being transcoded into compressed binary JData.  That removes a
full re-encode of the source corpus, keeps byte-level provenance with the
upstream dataset, and lets the store be a set of hardlinks costing no extra
space.  Headers are still parsed out and inlined, so the document remains
searchable.

**A document size budget.**  CouchDB enforces a maximum document size, and a
handful of datasets produce digests one to two orders of magnitude above it.
After conversion, oversized documents deterministically offload their largest
inline payloads to the store until they fit.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import re
import json
import math
import subprocess
import warnings

import numpy as np

from .njcas import CAS, annex_key, annex_key_info, cas_url

__all__ = [
    "bids2json",
    "dataset_version",
    "canonical_json",
    "fingerprint",
    "NJBIDS_DEFAULT",
]

NJBIDS_DEFAULT = {
    # inline size ceilings, in bytes of the *source* file
    "max_tsv": 1 << 20,
    "max_json": 1 << 20,
    "max_text": 1 << 20,
    "max_mat": 2 << 20,
    "max_bvec": 1 << 20,
    # element ceiling for inlining an array out of an HDF5/SNIRF container
    "max_h5_elem": 256,
    "max_snirf_elem": 256,
    # serialised size ceiling for the whole document
    "max_doc": 7_500_000,
    # subtrees carved out into their own document
    "split_dirs": ("derivatives",),
    # subtrees kept as link-only manifests (discoverable, but never inlined)
    "linkonly_dirs": ("sourcedata", "code", "stimuli"),
    "cas_url": None,
}

_TEXT_BASENAMES = ("README", "CHANGES", "LICENSE", "CITATION", "AUTHORS", "TASK")
_TEXT_EXT = (".md", ".txt", ".rst", ".m", ".cff", ".bib", ".tex")
_TABULAR_EXT = (".tsv", ".csv", ".tsv.gz", ".csv.gz")
_NIFTI_EXT = (".nii", ".nii.gz", ".hdr", ".img", ".img.gz")
_JSONISH_EXT = (".json", ".jmsh", ".jnii", ".jnirs", ".jgii", ".jbids")
_COMPOUND_EXT = (".nii.gz", ".tsv.gz", ".csv.gz", ".img.gz", ".gii.gz", ".mgh.gz")


# =============================================================================
# path / naming helpers
# =============================================================================


def fileext(path):
    """Extension of ``path``, honouring compound extensions such as ``.nii.gz``."""
    lower = path.lower()
    for ext in _COMPOUND_EXT:
        if lower.endswith(ext):
            return ext
    return os.path.splitext(lower)[1]


def _is_text_name(fname):
    stem = os.path.splitext(fname)[0].upper()
    return stem in _TEXT_BASENAMES or fname.upper() in _TEXT_BASENAMES


def _setpath(root, keys, value):
    """Insert ``value`` at the nested ``keys`` location, creating dicts on the way.

    Mirrors the ``jq setpath`` behaviour of the original shell pipeline, so a
    file at ``sub-01/anat/x.nii.gz`` lands at ``doc['sub-01']['anat']['x.nii.gz']``.
    """
    node = root
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value


def _getpath(root, keys):
    node = root
    for key in keys:
        node = node[key]
    return node


# =============================================================================
# deterministic serialisation
# =============================================================================


def _plain(data):
    """Recursively convert to JSON-serialisable builtins.

    Non-finite floats become None: BIDS tables encode missing values as ``n/a``,
    which the readers surface as NaN, and NaN is not valid JSON (CouchDB and
    Postgres jsonb both reject it).
    """
    if isinstance(data, np.ndarray):
        return _plain(data.tolist())
    if isinstance(data, dict):
        return {str(k): _plain(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_plain(v) for v in data]
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        val = float(data)
        return val if math.isfinite(val) else None
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    return data


def canonical_json(doc):
    """Serialise ``doc`` deterministically.

    Sorted keys and fixed separators mean identical input always yields
    identical bytes -- the property the whole versioning scheme rests on.  The
    original pipeline merged with shell globs, so its key order (and therefore
    its bytes) varied between runs.
    """
    return json.dumps(
        _plain(doc),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


_HASH_IN_URL = re.compile(r"hash=(sha256|sha1|md5):([0-9a-f]+)")


def _dehydrate(node):
    """Reduce every ``_DataLink_`` to its bare content hash.

    Used only for fingerprinting: it makes the fingerprint invariant to the
    server hostname, URL scheme and query decoration, so relocating the download
    endpoint later cannot invalidate a DOI that was already minted.
    """
    if isinstance(node, dict):
        out = {}
        for key, val in node.items():
            if key == "_DataLink_" and isinstance(val, str):
                match = _HASH_IN_URL.search(val)
                out[key] = "%s:%s" % (match.group(1), match.group(2)) if match else val
            else:
                out[key] = _dehydrate(val)
        return out
    if isinstance(node, list):
        return [_dehydrate(v) for v in node]
    return node


def fingerprint(doc, manifest):
    """Content fingerprint of a converted dataset version.

    Computed over a canonical manifest plus the URL-independent form of the
    document, rather than over the document bytes, so that it identifies the
    *dataset content* and not the encoding of its links.
    """
    import hashlib

    lines = [
        "%s\t%d\t%s" % (entry["sha256"], entry["size"], entry["path"])
        for entry in sorted(manifest, key=lambda e: e["path"])
    ]
    lines.append(
        "payload\t%s" % hashlib.sha256(canonical_json(_dehydrate(doc)).encode("utf-8")).hexdigest()
    )
    blob = "\n".join(lines) + "\n"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), blob


# =============================================================================
# version identity
# =============================================================================

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _git(dspath, *args):
    try:
        out = subprocess.run(
            ["git", "-C", dspath] + list(args),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def dataset_version(dspath, description=None):
    """Derive a stable version label and provenance for a dataset checkout.

    OpenNeuro tagging is not uniform -- some datasets carry semver snapshot tags,
    some carry legacy accession or ObjectId tags, and some carry none at all --
    so the label is *derived* rather than trusted, and the commit SHA is always
    recorded as the ground truth.

    Resolution order: a semver tag pointing at HEAD, then the ``vX.Y.Z`` suffix
    of ``DatasetDOI``, then ``commit-<short sha>``.
    """
    commit = _git(dspath, "rev-parse", "HEAD")
    info = {
        "SourceCommit": commit or None,
        "SourceRemote": _git(dspath, "config", "--get", "remote.origin.url") or None,
        "VersionSource": None,
    }

    semvers = []
    for tag in _git(dspath, "tag", "--points-at", "HEAD").splitlines():
        tag = tag.strip()
        match = _SEMVER.match(tag)
        if match:
            semvers.append((tuple(int(g) for g in match.groups()), tag))
    if semvers:
        info["Version"] = max(semvers)[1]
        info["VersionSource"] = "git-tag"
        return info

    doi = (description or {}).get("DatasetDOI") or ""
    match = re.search(r"\.v(\d+\.\d+\.\d+)$", str(doi))
    if match:
        info["Version"] = match.group(1)
        info["VersionSource"] = "dataset_description.DatasetDOI"
        info["DatasetDOI"] = str(doi)
        return info

    info["Version"] = "commit-%s" % (commit[:8] if commit else "unknown")
    info["VersionSource"] = "git-commit" if commit else "none"
    return info


def dataset_tags(dspath):
    """All semver snapshot tags in the checkout, oldest first."""
    tags = []
    for tag in _git(dspath, "tag").splitlines():
        match = _SEMVER.match(tag.strip())
        if match:
            tags.append((tuple(int(g) for g in match.groups()), tag.strip()))
    return [tag for _key, tag in sorted(tags)]


# =============================================================================
# per-file conversion
# =============================================================================


class _Converter:
    def __init__(self, dspath, dbname, dsname, cas, config):
        self.dspath = dspath
        self.dbname = dbname
        self.dsname = dsname
        self.cas = cas
        self.config = config
        self.manifest = []
        self.errors = []
        # doc-key path -> (source relpath, serialised length); candidates for
        # the size-budget offload pass
        self.inline = []
        self.counts = {}

    # -- helpers --------------------------------------------------------

    def _count(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _register(self, path, relpath, kind, store=True):
        """Record a file in the manifest, optionally materialising it in the store.

        Every file is manifested -- including the ones whose content is inlined
        in the document -- so the manifest is a complete content listing of the
        dataset version and the reported file/byte totals are accurate.  Inlined
        payloads are hashed but not materialised: creating a store object for
        every small sidecar would add millions of entries that nothing ever
        resolves.
        """
        key = annex_key(path)
        if os.path.islink(path) and not os.path.exists(path):
            # dangling annex symlink: content was never fetched, so the payload
            # cannot be hashed.  The annex key still carries the upstream size
            # and content hash, which is enough to reference the file.
            info = annex_key_info(key) or {}
            entry = {
                "path": relpath,
                "sha256": None,
                "size": info.get("size"),
                "kind": kind,
                "annexkey": key,
                "present": False,
            }
            self.manifest.append(entry)
            return entry
        if store:
            digest, size = self.cas.put(path, key=key)
        else:
            digest, size = self.cas.digest(path, key=key)
        entry = {
            "path": relpath,
            "sha256": digest,
            "size": size,
            "kind": kind,
            "annexkey": key,
            "present": True,
        }
        self.manifest.append(entry)
        return entry

    def _link(self, entry, jsonpath=None):
        """Build the ``_DataLink_`` node for a manifest entry."""
        if entry["sha256"] is None:
            # not fetched locally; fall back to an annex-key reference so the
            # link still identifies exact content and can be resolved once the
            # payload is retrieved
            info = annex_key_info(entry.get("annexkey")) or {}
            target = "annex:%s" % entry.get("annexkey") if entry.get("annexkey") else None
            if not target:
                return {"_DataLink_": "missing:%s" % entry["path"]}
            url = cas_url(
                info.get("hash", ""),
                size=info.get("size"),
                db=self.dbname,
                doc=self.dsname,
                file=entry["path"],
                base=self.config.get("cas_url"),
                algo=(info.get("backend", "MD5E")[:-1] or "md5").lower(),
            )
        else:
            url = cas_url(
                entry["sha256"],
                size=entry["size"],
                db=self.dbname,
                doc=self.dsname,
                file=entry["path"],
                base=self.config.get("cas_url"),
            )
        if jsonpath:
            url = url + ":" + jsonpath
        return {"_DataLink_": url}

    # -- dispatch -------------------------------------------------------

    def convert(self, path, relpath, linkonly=False):
        """Return the JSON value for one source file."""
        ext = fileext(relpath)
        fname = os.path.basename(relpath)
        dangling = os.path.islink(path) and not os.path.exists(path)

        if dangling:
            self._count("link-dangling")
            return self._link(self._register(path, relpath, "dangling"))

        try:
            size = os.path.getsize(path)
        except OSError as err:
            self.errors.append("%s: %s" % (relpath, err))
            return {"_DataLink_": "missing:%s" % relpath}

        if size == 0:
            self._count("empty")
            self._register(path, relpath, "empty", store=False)
            return {}

        if linkonly:
            self._count("link-only")
            return self._link(self._register(path, relpath, "linkonly"))

        try:
            if ext in _NIFTI_EXT:
                return self._nifti(path, relpath, size)
            if ext == ".gii":
                return self._gifti(path, relpath, size)
            if ext == ".snirf":
                return self._snirf(path, relpath, size)
            if ext in _TABULAR_EXT:
                return self._tabular(path, relpath, fname, size)
            if ext in _JSONISH_EXT:
                return self._jsonfile(path, relpath, size)
            if ext in (".bval", ".bvec"):
                return self._bvec(path, relpath, size)
            if ext in (".vhdr", ".vmrk"):
                return self._brainvision(path, relpath, size)
            if ext in (".edf", ".bdf"):
                return self._edf(path, relpath, size)
            if ext == ".set":
                return self._eeglab(path, relpath, size)
            if ext in (".nwb", ".h5", ".hdf5"):
                return self._hdf5(path, relpath, size)
            if ext == ".mat":
                return self._mat(path, relpath, size)
            if ext in _TEXT_EXT or _is_text_name(fname):
                return self._text(path, relpath, size)
        except Exception as err:  # a bad file must not abort the dataset
            self.errors.append("%s: %s: %s" % (relpath, type(err).__name__, err))
            warnings.warn("failed to parse %s: %s" % (relpath, err))

        self._count("link")
        return self._link(self._register(path, relpath, "binary"))

    # -- format handlers ------------------------------------------------

    def _text(self, path, relpath, size):
        if size > self.config["max_text"]:
            self._count("text-linked")
            return self._link(self._register(path, relpath, "text"))
        with open(path, "r", encoding="utf-8", errors="replace") as fid:
            text = fid.read()
        self._register(path, relpath, "text", store=False)
        self._count("text")
        return text

    def _jsonfile(self, path, relpath, size):
        """Parse a JSON sidecar, tolerating the two defects seen in practice.

        A byte-order mark makes ``json.load`` raise, so the file is decoded as
        utf-8-sig, which strips a BOM if present and is otherwise identical to
        utf-8.  A sidecar that is not valid JSON at all is kept verbatim as a
        string rather than reduced to a link: the content is still human
        readable and still searchable, which is the whole point of the digest,
        and the parse failure is recorded either way.
        """
        if size > self.config["max_json"]:
            self._count("json-linked")
            return self._link(self._register(path, relpath, "json"))
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fid:
            text = fid.read()
        self._register(path, relpath, "json", store=False)
        try:
            data = json.loads(text)
        except ValueError as err:
            self.errors.append("%s: invalid JSON kept as text: %s" % (relpath, err))
            self._count("json-malformed")
            return text
        self._count("json")
        return data

    def _tabular(self, path, relpath, fname, size):
        from .csvtsv import load_csv_tsv

        always = fname.startswith("participants.") or fname.endswith("_scans.tsv")
        if size > self.config["max_tsv"] and not always:
            self._count("tsv-linked")
            return self._link(self._register(path, relpath, "tabular"))
        delim = "," if fileext(fname).startswith(".csv") else "\t"
        data = load_csv_tsv(path, delimiter=delim, return_dict=True, convert_numeric=True)
        self._register(path, relpath, "tabular", store=False)
        self._count("tsv")
        return data

    def _bvec(self, path, relpath, size):
        if size > self.config["max_bvec"]:
            self._count("bvec-linked")
            return self._link(self._register(path, relpath, "bvec"))
        data = np.genfromtxt(path, dtype=np.float32)
        self._register(path, relpath, "bvec", store=False)
        self._count("bvec")
        return data

    def _nifti(self, path, relpath, size):
        """Inline the JNIfTI header; link the voxel payload to the store.

        Only the header is read: ``niiheader()`` inflates just the leading
        kilobyte, so this stays cheap even for a multi-gigabyte volume.
        """
        from .jnifti import niiheader, niiheader2jnii

        entry = self._register(path, relpath, "nifti")
        try:
            jnii = niiheader2jnii(niiheader(path))
        except Exception as err:
            self.errors.append("%s: nifti header: %s" % (relpath, err))
            self._count("nifti-headerfail")
            return self._link(entry)
        jnii = {k: v for k, v in jnii.items() if k != "NIFTIData"}
        jnii["NIFTIData"] = self._link(entry)
        self._count("nifti")
        return jnii

    def _gifti(self, path, relpath, size):
        from .jgifti import gii2jgii

        entry = self._register(path, relpath, "gifti")
        if size > self.config["max_json"]:
            jgii = {"GIFTIObject": self._link(entry)}
            self._count("gifti-linked")
            return jgii
        jgii = gii2jgii(path)
        self._count("gifti")
        return jgii

    def _snirf(self, path, relpath, size):
        """Inline SNIRF metadata; link the measurement arrays to the store."""
        from .njdigest import snirf_digest

        entry = self._register(path, relpath, "snirf")
        digest = snirf_digest(path, maxelem=self.config.get("max_snirf_elem", 4096))
        digest["SNIRFObject"] = self._link(entry)
        self._count("snirf")
        return digest

    def _eeglab(self, path, relpath, size):
        from .njdigest import eeglab_digest

        entry = self._register(path, relpath, "eeglab")
        digest = eeglab_digest(path, maxelem=self.config.get("max_h5_elem", 4096))
        digest["EEGLABObject"] = self._link(entry)
        self._count("eeglab")
        return digest

    def _hdf5(self, path, relpath, size):
        from .njdigest import hdf5_digest

        entry = self._register(path, relpath, "hdf5")
        digest = {
            "HDF5Data": hdf5_digest(path, maxelem=self.config.get("max_h5_elem", 4096)),
            "HDF5Object": self._link(entry),
        }
        self._count("hdf5")
        return digest

    def _mat(self, path, relpath, size):
        from .njdigest import mat_digest, eeglab_digest

        entry = self._register(path, relpath, "mat")
        with open(path, "rb") as fid:
            magic = fid.read(8)
        if magic == b"\x89HDF\r\n\x1a\n":
            from .njdigest import hdf5_digest

            digest = {"MATData": hdf5_digest(path, maxelem=self.config.get("max_h5_elem", 4096))}
        else:
            digest = {"MATData": mat_digest(path)}
        digest["MATObject"] = self._link(entry)
        self._count("mat")
        return digest

    def _brainvision(self, path, relpath, size):
        from .njdigest import bvheader

        self._register(path, relpath, "brainvision", store=False)
        self._count("brainvision")
        return bvheader(path)

    def _edf(self, path, relpath, size):
        from .njdigest import edfheader

        entry = self._register(path, relpath, "edf")
        digest = edfheader(path)
        digest["EDFObject"] = self._link(entry)
        self._count("edf")
        return digest


# =============================================================================
# dataset walk
# =============================================================================


def _walk(root, skip_hidden=True):
    """Deterministic recursive file listing, relative to ``root``."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not (skip_hidden and d.startswith(".")))
        for name in sorted(filenames):
            if skip_hidden and name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            out.append((full, os.path.relpath(full, root).replace(os.sep, "/")))
    return out


def _load_description(dspath):
    path = os.path.join(dspath, "dataset_description.json")
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fid:
            return json.load(fid)
    except Exception:
        return {}


def bids2json(dspath, dbname=None, dsname=None, cas=None, casroot=None, **kwargs):
    """Convert one BIDS dataset directory into version-invariant JSON documents.

    Parameters
    ----------
    dspath : str
        Dataset root (the directory holding ``dataset_description.json``).
    dbname, dsname : str
        Database and document names used in the ``_DataLink_`` decoration;
        default to the parent directory name and the dataset directory name.
    cas : jdata.njcas.CAS
        Store to register payloads in.  If omitted, one is created at
        ``casroot`` (or ``$NEUROJSON_CAS_ROOT``).

    Returns
    -------
    dict
        ``doc`` (main document), ``split`` (per-split-dir documents, e.g.
        derivatives), ``manifest``, ``version``, ``fingerprint``, ``stats`` and
        ``errors``.
    """
    config = dict(NJBIDS_DEFAULT)
    config.update({k: v for k, v in kwargs.items() if v is not None})

    dspath = os.path.abspath(dspath.rstrip("/"))
    dsname = dsname or os.path.basename(dspath)
    dbname = dbname or os.path.basename(os.path.dirname(dspath))

    if cas is None:
        casroot = casroot or os.environ.get("NEUROJSON_CAS_ROOT")
        if not casroot:
            raise ValueError("either cas or casroot (or $NEUROJSON_CAS_ROOT) is required")
        cas = CAS(casroot)

    description = _load_description(dspath)
    version = dataset_version(dspath, description)
    version["Tags"] = dataset_tags(dspath)

    conv = _Converter(dspath, dbname, dsname, cas, config)

    split_dirs = tuple(config["split_dirs"])
    linkonly_dirs = tuple(config["linkonly_dirs"])

    doc = {}
    split = {name: {} for name in split_dirs}

    for path, relpath in _walk(dspath):
        keys = relpath.split("/")
        top = keys[0]
        if top in split_dirs and len(keys) > 1:
            target, keypath = split[top], keys[1:]
            linkonly = True
        elif top in linkonly_dirs:
            target, keypath, linkonly = doc, keys, True
        else:
            target, keypath, linkonly = doc, keys, False

        value = conv.convert(path, relpath, linkonly=linkonly)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        _setpath(target, keypath, value)

        if not linkonly and not isinstance(value, dict):
            conv.inline.append((target is doc, tuple(keypath), relpath))
        elif not linkonly and isinstance(value, dict) and "_DataLink_" not in value:
            conv.inline.append((target is doc, tuple(keypath), relpath))

    # cross-reference the split-out documents from the main one
    for name in split_dirs:
        if split[name]:
            doc[name] = {
                "_DataLink_": "%s/%s_%s/%s"
                % (
                    (config.get("couch_url") or "").rstrip("/"),
                    dbname,
                    name.rstrip("s"),
                    dsname,
                )
                if config.get("couch_url")
                else "couch:%s_%s/%s" % (dbname, name.rstrip("s"), dsname)
            }

    stats = {"counts": dict(conv.counts), "files": len(conv.manifest)}
    doc, offloaded = _apply_budget(doc, conv, config)
    stats["offloaded"] = offloaded

    fpr, blob = fingerprint(doc, conv.manifest)
    doc[".neurojson"] = {
        "Version": version["Version"],
        "VersionSource": version["VersionSource"],
        "SourceCommit": version["SourceCommit"],
        "Fingerprint": fpr,
        "Tags": version["Tags"],
        "Files": len(conv.manifest),
        "Bytes": sum(e["size"] or 0 for e in conv.manifest),
    }
    if version.get("DatasetDOI"):
        doc[".neurojson"]["DatasetDOI"] = version["DatasetDOI"]

    return {
        "doc": doc,
        "split": {k: v for k, v in split.items() if v},
        "manifest": conv.manifest,
        "manifest_blob": blob,
        "version": version,
        "fingerprint": fpr,
        "stats": stats,
        "errors": conv.errors,
    }


def _apply_budget(doc, conv, config):
    """Offload the largest inline payloads until the document fits its budget.

    Candidates are ordered by ``(-serialised size, key path)`` so the choice is
    a deterministic function of the input: two runs over the same dataset
    offload exactly the same set, which is what keeps the fingerprint stable.
    """
    budget = config.get("max_doc") or 0
    offloaded = []
    if not budget:
        return doc, offloaded

    total = len(canonical_json(doc))
    if total <= budget:
        return doc, offloaded

    sized = []
    for is_main, keypath, relpath in conv.inline:
        if not is_main:
            continue
        try:
            node = _getpath(doc, list(keypath))
        except (KeyError, TypeError):
            continue
        sized.append((len(canonical_json(node)), keypath, relpath))
    sized.sort(key=lambda item: (-item[0], item[1]))

    for nodesize, keypath, relpath in sized:
        if total <= budget:
            break
        src = os.path.join(conv.dspath, relpath)
        if not os.path.isfile(src):
            continue
        entry = conv._register(src, relpath, "offloaded")
        conv.manifest.pop()  # already manifested during the initial pass
        link = conv._link(entry)
        _setpath(doc, list(keypath), link)
        total -= nodesize - len(canonical_json(link))
        offloaded.append({"path": relpath, "was": nodesize})

    return doc, offloaded
