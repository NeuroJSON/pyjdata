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

from .njcas import CAS, annex_key, annex_key_from_target, annex_key_info, cas_url

__all__ = [
    "bids2json",
    "strip_trailing_commas",
    "dataset_version",
    "canonical_json",
    "fingerprint",
    "NJBIDS_DEFAULT",
]

NJBIDS_DEFAULT = {
    # threads used to pre-hash annexed payloads within one dataset
    "hash_threads": 1,
    # most leaf payloads tier 1 will offload before handing over to tier 2
    "max_leaf_offloads": 256,
    # inline size ceilings, in bytes of the *source* file
    "max_tsv": 1 << 20,
    "max_json": 1 << 20,
    "max_text": 1 << 20,
    "max_mat": 2 << 20,
    "max_bvec": 1 << 20,
    # element ceiling for inlining an array out of an HDF5/SNIRF container
    "max_h5_elem": 256,
    "max_snirf_elem": 256,
    # Serialised size ceiling for the whole document.  Zero by default: a JSON
    # byte count is the wrong control, because CouchDB limits the *internal*
    # size of a parsed document and the ratio depends entirely on content.
    # Measured against CouchDB 3.4.2 with an 8 MB limit, the largest JSON
    # accepted was 4.19 MB for one big string, 7.23 MB for many short keys, and
    # over 29 MB for a float array -- a sevenfold spread. So documents are
    # converted whole and trimmed at publish time, when the server has actually
    # said no.
    "max_doc": 0,
    # subtrees carved out into their own document
    "split_dirs": ("derivatives",),
    # subtrees kept as link-only manifests (discoverable, but never inlined)
    "linkonly_dirs": ("sourcedata", "code", "stimuli"),
    "cas_url": None,
    # recorded in the metadata block so a document states which scheme produced
    # its identifiers and therefore its fingerprint
    "hash_algorithm": "sha256",
    "hash_source": "sha256",
    # re-encode modality payloads into binary JData attachments.  Empty means
    # reference the original file instead, which costs no read.
    "encode": (),
    "encode_codec": "zlib",
    # threads used inside the compressor for one attachment; zlib is
    # parallelised block-wise (see jdata.zlibmt) so this scales nearly linearly
    "encode_threads": 1,
    # do not re-encode a payload larger than this (0 = no limit)
    "max_encode": 0,
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


def strip_trailing_commas(text):
    """Remove commas that directly precede a closing brace or bracket.

    Trailing commas are invalid JSON but a common hand-editing artefact in BIDS
    sidecars, and rejecting the file costs every field in it.  The scan is
    string-aware: a comma inside a string literal is left alone, so a value
    like ``"a,]"`` cannot be silently corrupted.
    """
    out = []
    in_string = False
    escaped = False
    pending = []  # indices in `out` of commas that may yet turn out to be trailing
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            pending = []
            out.append(char)
            continue
        if char == ",":
            pending = [len(out)]
            out.append(char)
            continue
        if char in " \t\r\n":
            out.append(char)
            continue
        if char in "}]" and pending:
            out[pending[0]] = ""
        pending = []
        out.append(char)
    return "".join(out)


def _loads_tolerant(text):
    """Parse JSON, repairing trailing commas.  Returns ``(data, repair)``."""
    try:
        return json.loads(text), None
    except ValueError:
        repaired = strip_trailing_commas(text)
        if repaired != text:
            try:
                return json.loads(repaired), "trailing comma"
            except ValueError:
                pass
        raise


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

    # Each line names its algorithm.  Digests come from different algorithms in
    # one dataset -- sha256 for a re-encoded attachment's source, whatever the
    # annex key carries for a file that is only referenced -- so a bare hash
    # column would be ambiguous to anyone verifying it.
    lines = [
        "%s:%s\t%d\t%s"
        % (
            entry.get("algo") or "sha256",
            entry["sha256"],
            entry.get("size") or 0,
            entry["path"],
        )
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
        self.manifest_index = {}
        self.errors = []
        self._info = None
        # doc-key path -> (source relpath, serialised length); candidates for
        # the size-budget offload pass
        self.inline = []
        self.counts = {}

    # -- helpers --------------------------------------------------------

    def _count(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _register(self, path, relpath, kind, store=True, info=None):
        """Record a file in the manifest, optionally materialising it in the store.

        Every file is manifested -- including the ones whose content is inlined
        in the document -- so the manifest is a complete content listing of the
        dataset version and the reported file/byte totals are accurate.  Inlined
        payloads are hashed but not materialised: creating a store object for
        every small sidecar would add millions of entries that nothing ever
        resolves.
        """
        existing = self.manifest_index.get(relpath)
        if existing is not None:
            # A handler registers the file before parsing it, and a parse
            # failure falls through to the generic link branch, which would
            # register it a second time.  Two manifest lines for one path
            # inflate the file count and corrupt the fingerprint, so
            # registration is idempotent per path.
            if store and not existing.get("stored"):
                self.cas.identify(path, key=existing.get("annexkey") or annex_key(path), store=True)
                existing["stored"] = True
            return existing

        key = (
            annex_key_from_target(info.target)
            if info is not None and info.is_link
            else (None if info is not None else annex_key(path))
        )
        if (
            info.dangling
            if info is not None
            else (os.path.islink(path) and not os.path.exists(path))
        ):
            # dangling annex symlink: content was never fetched, so the payload
            # cannot be hashed.  The annex key still carries the upstream size
            # and content hash, which is enough to reference the file.
            info = annex_key_info(key) or {}
            entry = {
                "path": relpath,
                "algo": CAS.ANNEX_HASHES.get(info.get("backend", ""), "md5"),
                "sha256": None,
                # A dangling link that is not a git-annex key -- a broken
                # relative symlink, or one pointing outside the dataset -- has
                # no recoverable size, so record zero rather than None: the
                # manifest line is formatted numerically.
                "size": info.get("size") or 0,
                "kind": kind,
                "annexkey": key,
                "present": False,
            }
            self.manifest.append(entry)
            self.manifest_index[relpath] = entry
            return entry
        # identify() prefers whatever content hash the annex key already carries,
        # whichever backend produced it, and only reads the payload when there
        # is none to reuse
        ident = self.cas.identify(path, key=key, store=store)
        digest, size = ident["digest"], ident["size"]
        entry = {
            "path": relpath,
            "algo": ident["algo"],
            "sha256": digest,
            "size": size,
            "kind": kind,
            "annexkey": key,
            "present": True,
            "stored": bool(store),
        }
        self.manifest.append(entry)
        self.manifest_index[relpath] = entry
        return entry

    def _link(self, entry, jsonpath=None):
        """Build the ``_DataLink_`` node for a manifest entry.

        The algorithm is taken from the entry rather than assumed.  It used to
        be omitted here, so ``cas_url``'s default labelled every link
        ``sha256:`` -- including md5 digests taken from an annex key, which a
        verifier would then reject and the links view would report wrongly.
        """
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
                algo=entry.get("algo") or "md5",
            )
        else:
            url = cas_url(
                entry["sha256"],
                size=entry["size"],
                db=self.dbname,
                doc=self.dsname,
                file=entry["path"],
                base=self.config.get("cas_url"),
                algo=entry.get("algo") or self.cas.algo,
            )
        if jsonpath:
            url = url + ":" + jsonpath
        return {"_DataLink_": url}

    # -- dispatch -------------------------------------------------------

    def convert(self, info, linkonly=False):
        """Return the JSON value for one source file."""
        path, relpath = info.path, info.relpath
        ext = fileext(relpath)
        fname = os.path.basename(relpath)
        self._info = info

        if info.dangling:
            self._count("link-dangling")
            return self._link(self._register(path, relpath, "dangling", info=info))

        if not info.present:
            self.errors.append("%s: could not be stat()ed" % relpath)
            self._count("unreadable")
            return {"_DataLink_": "unreadable:%s" % relpath}

        size = info.size
        if size == 0:
            self._count("empty")
            self._register(path, relpath, "empty", store=False, info=info)
            return {}

        if linkonly:
            return self._safe_link(path, relpath, "linkonly", "link-only")

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

        return self._safe_link(path, relpath, "binary", "link")

    def _attachment_link(self, path, relpath, kind, ext, info, jsonpath=None):
        """Re-encode a payload as a binary JData attachment and link to it.

        Returns the ``_DataLink_`` node, or None if this file should be
        referenced as-is.  Deliberately does **not** build the inlined header:
        that stays the responsibility of the format handler, which uses the same
        header-only reader whether or not an attachment already exists.  Letting
        this method produce the header instead meant a cached attachment took a
        different code path from a fresh one and yielded a different document --
        and therefore a different fingerprint -- on the second run.

        The attachment is named ``<sha256 of the source>_<codec><ext>``.  Naming
        by the source rather than the encoded output keeps the name stable
        across a change of encoder or compression settings, and lets two dataset
        versions that share a file share one attachment.
        """
        from .njencode import encode_attachment, encoder_for, payload_digest

        spec = encoder_for(ext)
        if spec is None or not info.present:
            return None
        limit = int(self.config.get("max_encode") or 0)
        if limit and info.size > limit:
            self._count("encode-too-large")
            return None

        codec = self.config.get("encode_codec") or "zlib"
        entry = self._register(path, relpath, kind, store=False, info=info)

        # the attachment name is a sha256 of the source whatever the store's own
        # algorithm is, so compute it explicitly when they differ
        source = entry.get("sha256")
        if entry.get("algo") != "sha256" or not source:
            source, _size = self.cas.hashfile_with(path, "sha256")
            entry["sha256_source"] = source

        attach_ext = spec[0]
        suffix = ("_%s%s" % (codec, attach_ext)) if codec else attach_ext

        if self.cas.has(source, suffix):
            size = os.path.getsize(self.cas.objpath(source, suffix))
            self._count("encoded-cached")
        else:
            try:
                _header, payload, attach_ext, _keys = encode_attachment(
                    path,
                    ext,
                    compression=codec,
                    nthread=int(self.config.get("encode_threads") or 1),
                )
            except Exception as err:
                self.errors.append(
                    "%s: could not re-encode: %s: %s" % (relpath, type(err).__name__, err)
                )
                self._count("encode-failed")
                return None
            size, _created = self.cas.put_derived(source, payload, suffix)
            entry["attachment_sha256"] = payload_digest(payload)
            self._count("encoded")

        entry["attachment"] = suffix
        entry["attachment_size"] = size
        url = cas_url(
            source,
            size=size,
            db=self.dbname,
            doc=self.dsname,
            file=relpath,
            base=self.config.get("cas_url"),
            algo="sha256",
            enc=suffix,
        )
        if jsonpath:
            url += ":$." + jsonpath
        return {"_DataLink_": url}

    def _safe_link(self, path, relpath, kind, counter):
        """Register and link a file, tolerating an unreadable payload.

        This is the last resort for every file the specific handlers could not
        take, so it must not be able to raise: registering touches the payload
        (to hash it), and a file that cannot be read -- wrong permissions, a
        vanished network mount, a symlink loop -- would otherwise abort the
        whole dataset from inside the fallback that exists to prevent exactly
        that.
        """
        try:
            entry = self._register(path, relpath, kind, info=self._info)
        except OSError as err:
            self.errors.append("%s: unreadable: %s" % (relpath, err))
            self._count("unreadable")
            return {"_DataLink_": "unreadable:%s" % relpath}
        self._count(counter)
        return self._link(entry)

    # -- format handlers ------------------------------------------------

    def _text(self, path, relpath, size):
        if size > self.config["max_text"]:
            self._count("text-linked")
            return self._link(self._register(path, relpath, "text", info=self._info))
        with open(path, "r", encoding="utf-8", errors="replace") as fid:
            text = fid.read()
        self._register(path, relpath, "text", store=False, info=self._info)
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
            return self._link(self._register(path, relpath, "json", info=self._info))
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fid:
            text = fid.read()
        self._register(path, relpath, "json", store=False, info=self._info)
        if not text.strip():
            # a sidecar holding only whitespace is empty in intent; several
            # OpenNeuro datasets ship one-byte "\n" placeholders
            self._count("json-empty")
            return {}
        try:
            data, repair = _loads_tolerant(text)
        except ValueError as err:
            self.errors.append("%s: invalid JSON kept as text: %s" % (relpath, err))
            self._count("json-malformed")
            return text
        if repair:
            self.errors.append("%s: repaired invalid JSON (%s)" % (relpath, repair))
            self._count("json-repaired")
        else:
            self._count("json")
        return data

    def _tabular(self, path, relpath, fname, size):
        from .csvtsv import load_csv_tsv

        always = fname.startswith("participants.") or fname.endswith("_scans.tsv")
        if size > self.config["max_tsv"] and not always:
            self._count("tsv-linked")
            return self._link(self._register(path, relpath, "tabular", info=self._info))
        delim = "," if fileext(fname).startswith(".csv") else "\t"
        data = load_csv_tsv(path, delimiter=delim, return_dict=True, convert_numeric=True)
        self._register(path, relpath, "tabular", store=False, info=self._info)
        self._count("tsv")
        return data

    def _bvec(self, path, relpath, size):
        if size > self.config["max_bvec"]:
            self._count("bvec-linked")
            return self._link(self._register(path, relpath, "bvec", info=self._info))
        data = np.genfromtxt(path, dtype=np.float32)
        self._register(path, relpath, "bvec", store=False, info=self._info)
        self._count("bvec")
        return data

    def _nifti(self, path, relpath, size):
        """Inline the JNIfTI header; link the voxel payload to the store.

        Only the header is read: ``niiheader()`` inflates just the leading
        kilobyte, so this stays cheap even for a multi-gigabyte volume.
        """
        from .jnifti import niiheader, niiheader2jnii

        entry = self._register(path, relpath, "nifti", info=self._info)
        try:
            jnii = niiheader2jnii(niiheader(path))
        except Exception as err:
            self.errors.append("%s: nifti header: %s" % (relpath, err))
            self._count("nifti-headerfail")
            return self._link(entry)
        jnii = {k: v for k, v in jnii.items() if k != "NIFTIData"}
        attached = (
            self._attachment_link(
                path, relpath, "nifti", fileext(relpath), self._info, jsonpath="NIFTIData"
            )
            if "nii" in self.config.get("encode", ())
            else None
        )
        jnii["NIFTIData"] = attached if attached is not None else self._link(entry)
        self._count("nifti")
        return jnii

    def _gifti(self, path, relpath, size):
        from .jgifti import gii2jgii

        entry = self._register(path, relpath, "gifti", info=self._info)
        attached = (
            self._attachment_link(path, relpath, "gifti", ".gii", self._info)
            if "gii" in self.config.get("encode", ())
            else None
        )
        if attached is not None:
            self._count("gifti-encoded")
            return {"GIFTIObject": attached}
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

        entry = self._register(path, relpath, "snirf", info=self._info)
        digest = snirf_digest(path, maxelem=self.config.get("max_snirf_elem", 4096))
        digest["SNIRFObject"] = self._link(entry)
        self._count("snirf")
        return digest

    def _eeglab(self, path, relpath, size):
        from .njdigest import eeglab_digest

        entry = self._register(path, relpath, "eeglab", info=self._info)
        digest = eeglab_digest(path, maxelem=self.config.get("max_h5_elem", 4096))
        digest["EEGLABObject"] = self._link(entry)
        self._count("eeglab")
        return digest

    def _hdf5(self, path, relpath, size):
        from .njdigest import hdf5_digest

        entry = self._register(path, relpath, "hdf5", info=self._info)
        digest = {
            "HDF5Data": hdf5_digest(path, maxelem=self.config.get("max_h5_elem", 4096)),
            "HDF5Object": self._link(entry),
        }
        self._count("hdf5")
        return digest

    def _mat(self, path, relpath, size):
        """Digest a ``.mat`` file, which is not always a MATLAB file.

        FSL and TBSS write plain-text VEST design matrices with a ``.mat``
        extension; ds006391 alone has 98 of them.  Handing those to scipy raises
        "Unknown mat file type", which previously produced an error per file and
        discarded metadata that is perfectly readable.
        """
        from .njdigest import hdf5_digest, is_matfile, mat_digest, vest_header

        entry = self._register(path, relpath, "mat", info=self._info)
        if not is_matfile(path):
            vest = vest_header(path)
            if vest:
                vest["MATObject"] = self._link(entry)
                self._count("vest")
                return vest
            self._count("mat-unrecognised")
            return self._link(entry)

        with open(path, "rb") as fid:
            magic = fid.read(8)
        if magic == b"\x89HDF\r\n\x1a\n":
            digest = {"MATData": hdf5_digest(path, maxelem=self.config.get("max_h5_elem", 256))}
        else:
            digest = {"MATData": mat_digest(path)}
        digest["MATObject"] = self._link(entry)
        self._count("mat")
        return digest

    def _brainvision(self, path, relpath, size):
        from .njdigest import bvheader

        self._register(path, relpath, "brainvision", store=False, info=self._info)
        self._count("brainvision")
        return bvheader(path)

    def _edf(self, path, relpath, size):
        from .njdigest import edfheader

        entry = self._register(path, relpath, "edf", info=self._info)
        digest = edfheader(path)
        digest["EDFObject"] = self._link(entry)
        self._count("edf")
        return digest


# =============================================================================
# dataset walk
# =============================================================================


class FileInfo:
    """What one directory entry is, gathered once.

    A conversion pass over this corpus makes tens of millions of metadata calls,
    and on spinning disks that -- not payload bandwidth -- is the wall.  The
    naive sequence per file was ``islink`` + ``exists`` + ``getsize`` +
    ``islink`` + ``readlink``: five syscalls, four of which re-ask the kernel
    something the directory read already answered.  ``os.scandir`` hands back an
    entry whose type comes free from the directory block and whose ``stat`` is
    cached, so a regular file now costs one ``lstat`` and a symlink one
    ``lstat`` plus a ``readlink``.
    """

    __slots__ = ("path", "relpath", "is_link", "present", "size", "target")

    def __init__(self, path, relpath, is_link, present, size, target):
        self.path = path
        self.relpath = relpath
        self.is_link = is_link
        self.present = present
        self.size = size
        self.target = target

    @property
    def dangling(self):
        return self.is_link and not self.present


def _walk(root, skip_hidden=True):
    """Deterministic recursive file listing, relative to ``root``.

    Returns :class:`FileInfo` objects.  Ordering is by sorted directory name
    then sorted file name at every level, so the document is built in the same
    order on every run.
    """
    out = []

    def visit(dirpath):
        try:
            with os.scandir(dirpath) as entries:
                items = list(entries)
        except OSError:
            return
        dirs, files = [], []
        for entry in items:
            if skip_hidden and entry.name.startswith("."):
                continue
            try:
                isdir = entry.is_dir(follow_symlinks=False)
            except OSError:
                isdir = False
            (dirs if isdir else files).append(entry)
        for entry in sorted(files, key=lambda e: e.name):
            out.append(_info(entry, root))
        for entry in sorted(dirs, key=lambda e: e.name):
            visit(entry.path)

    visit(root)
    return out


def _info(entry, root):
    relpath = os.path.relpath(entry.path, root).replace(os.sep, "/")
    try:
        is_link = entry.is_symlink()
    except OSError:
        is_link = False
    target = None
    present = True
    size = 0
    if is_link:
        try:
            target = os.readlink(entry.path)
        except OSError:
            target = None
        try:
            size = entry.stat(follow_symlinks=True).st_size
        except OSError:
            present = False
    else:
        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            present = False
    return FileInfo(entry.path, relpath, is_link, present, size, target)


#: extensions whose contents get inlined, and therefore have to be read
_INLINE_EXT = frozenset(
    _TABULAR_EXT + _JSONISH_EXT + _TEXT_EXT + (".bval", ".bvec", ".vhdr", ".vmrk")
)


def _prefetch(files, cas, threads, config):
    """Warm and register a dataset's files in parallel, before the walk.

    Two costs are overlapped here, and both are latency rather than bandwidth:

    * **Small-file reads.**  Inlining metadata means opening every ``.json`` and
      ``.tsv`` sidecar, and on spinning disks each of those is an independent
      seek.  Measured on ds002785, 4290 sidecars accounted for 57 of the
      dataset's 61 seconds -- about 13 ms each, which is seek time, not work.
      Reading them concurrently lets the queue depth hide the latency; the data
      lands in the filesystem cache and the sequential pass then hits it warm.
    * **Store registration** for annexed payloads.

    Parallelism in the pipeline is otherwise per dataset, which is the wrong
    granularity for the largest ones: they hold six figures of files, so one
    worker walks them serially while the rest of the pool idles.

    Determinism is unaffected.  This only warms the cache and populates the
    store; the document is still built by the ordered sequential walk.
    """
    from concurrent.futures import ThreadPoolExecutor

    ceiling = max(
        int(config.get("max_json") or 0),
        int(config.get("max_tsv") or 0),
        int(config.get("max_text") or 0),
    )

    register, warm = [], []
    for info in files:
        if info.dangling:
            continue  # content never fetched; nothing to read or hash
        if info.is_link:
            key = annex_key_from_target(info.target)
            if key:
                register.append((info.path, key))
                continue
        if 0 < info.size <= ceiling and fileext(info.relpath) in _INLINE_EXT:
            warm.append(info.path)
    if not (register or warm):
        return 0

    def do_register(item):
        path, key = item
        try:
            cas.identify(path, key=key, store=True)
        except OSError:
            pass

    def do_warm(path):
        # the read is for its side effect on the filesystem cache
        try:
            with open(path, "rb") as fid:
                fid.read(1 << 20)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(do_register, register))
        list(pool.map(do_warm, warm))
    cas.flush()
    return len(register) + len(warm)


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

    files = _walk(dspath)
    hash_threads = int(config.get("hash_threads") or 1)
    if hash_threads > 1:
        _prefetch(files, cas, hash_threads, config)

    doc = {}
    split = {name: {} for name in split_dirs}

    for info in files:
        path, relpath = info.path, info.relpath
        keys = relpath.split("/")
        top = keys[0]
        if top in split_dirs and len(keys) > 1:
            target, keypath = split[top], keys[1:]
            linkonly = True
        elif top in linkonly_dirs:
            target, keypath, linkonly = doc, keys, True
        else:
            target, keypath, linkonly = doc, keys, False

        value = conv.convert(info, linkonly=linkonly)
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

    # split documents are published too, so they get the same budget.  Their
    # subtrees are already link-only, so in practice tier 2 does the work: a
    # 380k-file derivatives tree is past the budget on links alone.
    split_offloaded = {}
    for name in list(split):
        if not split[name]:
            continue
        split[name], dropped = _apply_budget(split[name], conv, config, protected=(), label=name)
        if dropped:
            split_offloaded[name] = dropped
    if split_offloaded:
        stats["split_offloaded"] = split_offloaded

    fpr, blob = fingerprint(doc, conv.manifest)
    doc[".neurojson"] = {
        "Version": version["Version"],
        "VersionSource": version["VersionSource"],
        "SourceCommit": version["SourceCommit"],
        "Fingerprint": fpr,
        "Tags": version["Tags"],
        "Files": len(conv.manifest),
        "Bytes": sum(e["size"] or 0 for e in conv.manifest),
        "HashAlgorithm": cas.algo,
        "HashSource": config.get("hash_source") or cas.algo,
    }
    if config.get("encode"):
        doc[".neurojson"]["Encoded"] = sorted(config["encode"])
        doc[".neurojson"]["EncodeCodec"] = config.get("encode_codec") or "zlib"
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


#: never offloaded: these carry the dataset-level metadata that makes a document
#: discoverable and its participants searchable at all
BUDGET_PROTECTED = (
    ".neurojson",
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "README.md",
    "README.rst",
    "CHANGES",
    "CITATION.cff",
    "LICENSE",
)


def _offload_node(doc, keypath, conv, label):
    """Replace a document subtree with a link to it, stored in the CAS."""
    node = _getpath(doc, list(keypath))
    payload = canonical_json(node)
    digest, size = conv.cas.put_bytes(payload)
    link = {
        "_DataLink_": cas_url(
            digest,
            size=size,
            db=conv.dbname,
            doc=conv.dsname,
            file="%s.json" % "/".join(keypath),
            base=conv.config.get("cas_url"),
        )
    }
    _setpath(doc, list(keypath), link)
    return link, len(payload), {"path": "/".join(keypath), "was": len(payload), "how": label}


def _apply_budget(doc, conv, config, protected=BUDGET_PROTECTED, label="main"):
    """Shrink a document to its size budget, deterministically.

    Two tiers, applied in order:

    1. inline leaf payloads -- a large table or sidecar whose content is in the
       document -- are replaced by a link to the original file;
    2. whole top-level subtrees (a subject directory, or a derivative pipeline)
       are serialised into the store and replaced by a link.

    The second tier exists because the first cannot always be enough: a dataset
    of 177k files is past the budget even when every single file is already
    nothing but a link, and no amount of leaf offloading helps.  Something has
    to give, so what gives is per-file detail for the largest subtrees, while
    the dataset-level metadata that makes the document findable -- description,
    participants table, README -- is never touched.

    Candidates are ordered by ``(-serialised size, key)`` in both tiers, so the
    set that gets offloaded is a function of the input alone and the resulting
    fingerprint stays reproducible.
    """
    budget = config.get("max_doc") or 0
    offloaded = []
    if not budget:
        return doc, offloaded

    total = len(canonical_json(doc))
    if total <= budget:
        return doc, offloaded

    # tier 1: inline leaf payloads, replaced by a link to the original file
    sized = []
    for is_main, keypath, relpath in conv.inline:
        if (label == "main") != bool(is_main):
            continue
        if keypath and keypath[0] in protected:
            continue
        try:
            node = _getpath(doc, list(keypath))
        except (KeyError, TypeError):
            continue
        sized.append((len(canonical_json(node)), keypath, relpath))
    sized.sort(key=lambda item: (-item[0], item[1]))

    # Tier 1 is capped.  Offloading leaves one at a time is the right move for a
    # handful of oversized tables, but on a very wide dataset it degenerates:
    # ds004186 needed 121993 leaf offloads, which took 34 minutes and created
    # 122k tiny store objects, where shedding a few subtrees would have done it
    # in seconds.  Past the cap, hand over to tier 2.
    cap = int(config.get("max_leaf_offloads") or 0) or len(sized)
    for nodesize, keypath, relpath in sized[:cap]:
        if total <= budget:
            break
        src = os.path.join(conv.dspath, relpath)
        if not os.path.isfile(src):
            continue
        entry = conv._register(src, relpath, "offloaded")
        link = conv._link(entry)
        _setpath(doc, list(keypath), link)
        total -= nodesize - len(canonical_json(link))
        offloaded.append({"path": relpath, "was": nodesize, "how": "leaf"})

    # The running total is an estimate: replacing a node also shifts separators
    # and key ordering, so it drifts from the real serialised length.  Re-measure
    # before deciding whether tier 2 is needed, otherwise the budget is not
    # actually enforced -- ds003097 came out 211 bytes over.
    total = len(canonical_json(doc))
    if total <= budget:
        return doc, offloaded

    # tier 2: whole top-level subtrees, serialised into the store
    subtrees = []
    for key, value in doc.items():
        if key in protected or not isinstance(value, dict):
            continue
        if "_DataLink_" in value:
            continue
        subtrees.append((len(canonical_json(value)), key))
    subtrees.sort(key=lambda item: (-item[0], item[1]))

    for nodesize, key in subtrees:
        if total <= budget:
            break
        link, was, record = _offload_node(doc, (key,), conv, "subtree")
        offloaded.append(record)
        total = len(canonical_json(doc))

    return doc, offloaded
