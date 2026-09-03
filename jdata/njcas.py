"""
Content-addressed store (CAS) for NeuroJSON dataset attachments.

Large, non-searchable payloads (NIfTI volumes, SNIRF/EEG recordings, images,
archives) are not embedded in the JSON digest and are not re-encoded.  They are
registered in a content-addressed store keyed by the SHA-256 of the *original*
file bytes, and referenced from the digest through an immutable ``_DataLink_``
URL.  Because the identifier is derived from the content, a given URL can never
start resolving to different bytes -- which is what makes a per-version DOI of a
dataset meaningful.

Layout::

    <casroot>/objects/<h[0:2]>/<h[2:4]>/<h>     the payload (hardlink by default)
    <casroot>/index.sqlite                      hash memo, keyed by git-annex key
    <casroot>/index/                             (reserved)

Two properties keep the cost down on a datalad/git-annex mirror such as
OpenNeuro:

* annex objects are already content-addressed (``MD5E-s<size>--<md5>.<ext>``),
  so the SHA-256 memo can be keyed by the annex key.  A file that is
  byte-identical across dataset versions is therefore hashed once, ever.
* annex objects live on the same filesystem as the store, so ``os.link()``
  succeeds and the store costs no extra space.  The hardlink additionally keeps
  the bytes alive if ``git annex drop`` later unlinks the annex copy.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import re
import sqlite3
import hashlib
import threading
import urllib.parse

__all__ = [
    "CAS",
    "annex_key",
    "annex_key_from_target",
    "annex_key_info",
    "cas_url",
    "DEFAULT_CAS_URL",
]

DEFAULT_CAS_URL = os.environ.get("NEUROJSON_CAS_URL", "https://neurojson.io/io/cas.cgi?action=get")

# git-annex keys look like  MD5E-s5663237--4608ffbd6b78ce3a325eb338fa556589.nii.gz
# (BACKEND[-sSIZE][-mMTIME]--HASH[.EXT]); SHA256E/SHA1E/URL backends are also seen.
_ANNEX_RE = re.compile(r"(?:^|/)\.git/annex/objects/(?:[^/]+/){2}(?P<key>[^/]+)/(?P=key)$")
_ANNEX_KEY_RE = re.compile(
    r"^(?P<backend>[A-Z0-9]+)(?P<fields>(?:-[a-zA-Z]\d+)*)--(?P<hash>[^.]+)(?P<ext>\..*)?$"
)

_READ_CHUNK = 4 << 20  # 4 MiB: large enough that ZFS readahead stays useful

#: expected hex digest length per algorithm, used to sanity-check an annex key
_HEX_LENGTHS = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}


def annex_key_from_target(target):
    """Return the git-annex key from a symlink target string, or None.

    Split out from :func:`annex_key` so a caller that has already read the link
    -- the directory walk does -- need not read it again.
    """
    if not target:
        return None
    match = _ANNEX_RE.search(target.replace(os.sep, "/"))
    return match.group("key") if match else None


def annex_key(path):
    """Return the git-annex key a symlink points at, or None.

    Works for both present and dangling annex symlinks, since only the link
    target text is inspected -- no I/O on the payload.
    """
    if not os.path.islink(path):
        return None
    return annex_key_from_target(os.readlink(path))


def annex_key_info(key):
    """Parse a git-annex key into ``{backend, size, hash, ext}``.

    ``size`` is None when the backend does not record it (e.g. plain ``MD5``).
    """
    if not key:
        return None
    match = _ANNEX_KEY_RE.match(key)
    if not match:
        return None
    size = None
    for field in re.findall(r"-([a-zA-Z])(\d+)", match.group("fields") or ""):
        if field[0] == "s":
            size = int(field[1])
    return {
        "backend": match.group("backend"),
        "size": size,
        "hash": match.group("hash"),
        "ext": match.group("ext") or "",
    }


def cas_url(digest, size=None, db=None, doc=None, file=None, base=None, algo="sha256", enc=None):
    """Build the immutable ``_DataLink_`` URL for a stored object.

    Only ``hash`` is authoritative.  ``db``/``doc``/``file``/``size`` are
    human-readable decoration, and are also what keeps existing NeuroJSON
    consumers working: ``jdata.jfile.jsoncache`` derives its download cache path
    from ``db``/``doc``, the CouchDB ``links`` view parses ``file=`` and
    ``size=``, and the web frontend shows ``size=`` before downloading.

    The parameter order is fixed so that the same inputs always produce the same
    string, byte for byte.
    """
    parts = [base or DEFAULT_CAS_URL]
    if db:
        parts.append("db=" + urllib.parse.quote(str(db), safe=""))
    if doc:
        parts.append("doc=" + urllib.parse.quote(str(doc), safe=""))
    parts.append("hash=%s:%s" % (algo, digest))
    if enc:
        # names the derived encoding stored beside the source content
        parts.append("enc=" + urllib.parse.quote(str(enc), safe="._"))
    if size is not None:
        parts.append("size=%d" % int(size))
    if file:
        parts.append("file=" + urllib.parse.quote(str(file), safe="/"))
    return parts[0] + ("&" if "?" in parts[0] else "?") + "&".join(parts[1:])


class CAS:
    """A content-addressed store with a git-annex aware hash memo.

    Parameters
    ----------
    root : str
        Store root directory; created on demand.
    algo : str
        Hash algorithm for object names (default ``sha256``).
    mode : str
        How an object is materialised: ``link`` (hardlink, default),
        ``symlink``, ``copy``, or ``none`` (compute and memoise the hash but do
        not materialise -- useful when the payload is served from elsewhere).
    memo : bool
        Use the SQLite hash memo (default True).
    """

    #: git-annex backends whose key already contains a hash of the payload,
    #: mapped to the hash algorithm that produced it
    ANNEX_HASHES = {
        "MD5": "md5",
        "MD5E": "md5",
        "SHA1": "sha1",
        "SHA1E": "sha1",
        "SHA256": "sha256",
        "SHA256E": "sha256",
        "SHA512": "sha512",
        "SHA512E": "sha512",
    }

    def __init__(
        self,
        root,
        algo="sha256",
        mode="link",
        memo=True,
        commit_every=64,
        busy_timeout=120.0,
        annex_hash=False,
    ):
        if mode not in ("link", "symlink", "copy", "none"):
            raise ValueError("mode must be link, symlink, copy or none")
        self.root = os.path.abspath(root)
        self.algo = algo
        self.mode = mode
        self.objroot = os.path.join(self.root, "objects")
        self.dbpath = os.path.join(self.root, "index.sqlite")
        self._use_memo = memo
        self.commit_every = max(1, int(commit_every))
        self.busy_timeout = float(busy_timeout)
        # Take the content hash from the git-annex key instead of reading the
        # payload.  An MD5E key such as MD5E-s5663237--4608ff... already states
        # the content hash and the exact size, so this is free -- and the read
        # needed to hash is otherwise the entire runtime of a pass over a
        # multi-terabyte mirror.
        self.annex_hash = bool(annex_hash)
        self._local = threading.local()
        self.stats = {
            "hashed": 0,
            "memo_hits": 0,
            "bytes_hashed": 0,
            "linked": 0,
            "already": 0,
            "memo_errors": 0,
            "from_annex": 0,
            "encoded": 0,
            "encoded_bytes": 0,
        }
        self._statlock = threading.Lock()
        os.makedirs(self.objroot, exist_ok=True)
        if self._use_memo and not self._init_memo():
            # The memo is only ever an optimisation.  Creating its schema takes
            # an exclusive lock, so a pool of workers opening a brand-new store
            # at the same moment can collide; that must degrade to "no memo",
            # not fail the worker before it has converted anything.
            self._use_memo = False
            with self._statlock:
                self.stats["memo_errors"] += 1

    # -- memo ---------------------------------------------------------------

    def _init_memo(self):
        # journal_mode is a persistent property of the database file, so it is
        # set once, here.  Re-issuing it on every connection needs an exclusive
        # lock, and with a wide process pool all opening the memo at once that
        # lock is what actually serialises the pool.
        conn = None
        try:
            conn = sqlite3.connect(self.dbpath, timeout=self.busy_timeout)
            conn.execute("PRAGMA busy_timeout=%d" % int(self.busy_timeout * 1000))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS objects ("
                " key TEXT PRIMARY KEY,"
                " algo TEXT NOT NULL,"
                " digest TEXT NOT NULL,"
                " size INTEGER NOT NULL)"
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    @property
    def _conn(self):
        """One SQLite connection per thread (sqlite3 objects are not shareable)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.dbpath, timeout=self.busy_timeout)
            conn.execute("PRAGMA busy_timeout=%d" % int(self.busy_timeout * 1000))
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def memo_get(self, key):
        """Look up a cached hash; any memo failure degrades to a cache miss.

        The memo is only ever an optimisation, so no database condition -- lock
        contention, a corrupt file, a read-only mount -- may be allowed to fail
        a conversion.  The cost of swallowing the error is one re-hash.
        """
        if not (self._use_memo and key):
            return None
        try:
            row = self._conn.execute(
                "SELECT digest, size FROM objects WHERE key=? AND algo=?",
                (key, self.algo),
            ).fetchone()
        except sqlite3.Error:
            with self._statlock:
                self.stats["memo_errors"] += 1
            return None
        return (row[0], row[1]) if row else None

    def memo_put(self, key, digest, size):
        """Record a hash in the memo, committing in batches.

        SQLite in WAL mode allows one writer at a time, so committing on every
        insert would serialise a wide parallel conversion on the memo rather
        than on I/O.  Batching keeps that cost negligible; the memo is a cache,
        so losing the tail of a batch to a crash only costs a re-hash.
        """
        if not (self._use_memo and key):
            return
        try:
            conn = self._conn
            conn.execute(
                "INSERT OR REPLACE INTO objects (key, algo, digest, size) VALUES (?,?,?,?)",
                (key, self.algo, digest, int(size)),
            )
            pending = getattr(self._local, "pending", 0) + 1
            if pending >= self.commit_every:
                conn.commit()
                pending = 0
            self._local.pending = pending
        except sqlite3.Error:
            with self._statlock:
                self.stats["memo_errors"] += 1

    def flush(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None and getattr(self._local, "pending", 0):
            try:
                conn.commit()
            except sqlite3.Error:
                with self._statlock:
                    self.stats["memo_errors"] += 1
            self._local.pending = 0

    def memo_count(self):
        if not self._use_memo:
            return 0
        try:
            return self._conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        except sqlite3.Error:
            return -1

    # -- object paths -------------------------------------------------------

    def objpath(self, digest, suffix=""):
        """Path of an object, optionally of a named derivative of it.

        ``suffix`` names a *derived* encoding of the same content -- e.g.
        ``_zlib.bnii`` for the binary JData re-encoding of a NIfTI volume.  The
        derivative sits beside the original under the source content's digest,
        so it can be found from the source alone and two dataset versions
        sharing a file share its attachment too.
        """
        return os.path.join(self.objroot, digest[0:2], digest[2:4], digest + suffix)

    def has(self, digest, suffix=""):
        return os.path.exists(self.objpath(digest, suffix))

    def put_derived(self, digest, payload, suffix):
        """Store a derived encoding named ``<digest><suffix>``.

        Returns ``(size, created)``.  Idempotent: an attachment already present
        is left alone, so re-converting an unchanged dataset re-encodes nothing.
        """
        dest = self.objpath(digest, suffix)
        if os.path.exists(dest):
            with self._statlock:
                self.stats["already"] += 1
            return os.path.getsize(dest), False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp.%d.%d" % (os.getpid(), threading.get_ident())
        try:
            with open(tmp, "wb") as fid:
                fid.write(payload)
            os.replace(tmp, dest)
            with self._statlock:
                self.stats["encoded"] += 1
                self.stats["encoded_bytes"] += len(payload)
        except FileExistsError:
            pass
        finally:
            if os.path.lexists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return len(payload), True

    def hashfile_with(self, path, algo):
        """Stream-hash a file with an explicit algorithm."""
        engine = hashlib.new(algo)
        size = 0
        with open(path, "rb") as fid:
            while True:
                chunk = fid.read(_READ_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                engine.update(chunk)
        with self._statlock:
            self.stats["hashed"] += 1
            self.stats["bytes_hashed"] += size
        return engine.hexdigest(), size

    # -- hashing ------------------------------------------------------------

    def hashfile(self, path):
        """Stream-hash a file, returning ``(digest, size)``."""
        h = hashlib.new(self.algo)
        size = 0
        with open(path, "rb") as fid:
            while True:
                chunk = fid.read(_READ_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
        with self._statlock:
            self.stats["hashed"] += 1
            self.stats["bytes_hashed"] += size
        return h.hexdigest(), size

    def annex_digest(self, key):
        """Return ``(digest, size)`` straight from a git-annex key, or None.

        Only accepted when the key's backend hash matches this store's algorithm,
        so one identifier scheme covers the whole corpus.
        """
        info = annex_key_info(key)
        if not info or info.get("size") is None:
            return None
        if self.ANNEX_HASHES.get(info["backend"]) != self.algo:
            return None
        digest = info["hash"].lower()
        if len(digest) != _HEX_LENGTHS.get(self.algo, 0):
            return None
        if any(ch not in "0123456789abcdef" for ch in digest):
            return None
        return digest, int(info["size"])

    def digest(self, path, key=None):
        """Return ``(digest, size)`` for a file, consulting the memo first.

        ``key`` overrides the auto-detected git-annex key.  Files that are not
        annex-managed are hashed directly and not memoised (they are small
        text/JSON/TSV payloads, so re-hashing is cheap and the memo stays small).
        """
        if key is None:
            key = annex_key(path)
        if self.annex_hash and key:
            fromkey = self.annex_digest(key)
            if fromkey:
                with self._statlock:
                    self.stats["from_annex"] += 1
                return fromkey
        cached = self.memo_get(key)
        if cached:
            with self._statlock:
                self.stats["memo_hits"] += 1
            return cached
        digest, size = self.hashfile(path)
        if key:
            self.memo_put(key, digest, size)
        return digest, size

    # -- store --------------------------------------------------------------

    def put(self, path, key=None):
        """Register a file in the store; returns ``(digest, size)``.

        Idempotent: an object already present is left untouched, so a re-run
        over an unchanged dataset performs no writes at all.
        """
        digest, size = self.digest(path, key=key)
        if self.mode == "none":
            return digest, size
        dest = self.objpath(digest)
        if os.path.exists(dest):
            with self._statlock:
                self.stats["already"] += 1
            return digest, size
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = os.path.realpath(path)
        tmp = dest + ".tmp.%d.%d" % (os.getpid(), threading.get_ident())
        try:
            if self.mode == "link":
                try:
                    os.link(src, tmp)
                except OSError:
                    # different filesystem, or link count limit -- fall back to
                    # a copy rather than silently losing the object
                    _copyfile(src, tmp)
            elif self.mode == "symlink":
                os.symlink(src, tmp)
            else:
                _copyfile(src, tmp)
            os.replace(tmp, dest)
            with self._statlock:
                self.stats["linked"] += 1
        except FileExistsError:
            pass
        finally:
            if os.path.lexists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return digest, size

    def put_bytes(self, data):
        """Register an in-memory payload; returns ``(digest, size)``.

        Used for content the pipeline generates rather than finds on disk --
        an offloaded subtree of a document, or a manifest.  Such an object is
        necessarily a real file rather than a hardlink, but they are small and
        few compared with the corpus.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.new(self.algo, data).hexdigest()
        size = len(data)
        with self._statlock:
            self.stats["hashed"] += 1
            self.stats["bytes_hashed"] += size
        if self.mode == "none":
            return digest, size
        dest = self.objpath(digest)
        if os.path.exists(dest):
            with self._statlock:
                self.stats["already"] += 1
            return digest, size
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp.%d.%d" % (os.getpid(), threading.get_ident())
        try:
            with open(tmp, "wb") as fid:
                fid.write(data)
            os.replace(tmp, dest)
            with self._statlock:
                self.stats["linked"] += 1
        except FileExistsError:
            pass
        finally:
            if os.path.lexists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return digest, size

    def url(self, digest, size=None, db=None, doc=None, file=None, base=None):
        return cas_url(digest, size=size, db=db, doc=doc, file=file, base=base, algo=self.algo)

    # -- maintenance --------------------------------------------------------

    def iterobjects(self):
        for dirpath, _dirs, files in os.walk(self.objroot):
            for name in files:
                if not name.endswith(".tmp") and ".tmp." not in name:
                    yield os.path.join(dirpath, name)

    def sample_objects(self, count, seed=0):
        """Return up to ``count`` object paths without walking the whole store.

        A full ``os.walk`` of the object tree costs one directory read per
        two-level prefix, which for a store holding millions of objects is far
        more work than the check itself.  Sampling random prefixes gives a
        usable spot check at a cost that does not grow with the store.
        """
        import random

        rng = random.Random(seed)
        try:
            outer = sorted(os.listdir(self.objroot))
        except OSError:
            return []
        rng.shuffle(outer)
        picked = []
        for first in outer:
            firstdir = os.path.join(self.objroot, first)
            try:
                inner = sorted(os.listdir(firstdir))
            except OSError:
                continue
            rng.shuffle(inner)
            for second in inner:
                seconddir = os.path.join(firstdir, second)
                try:
                    names = [n for n in os.listdir(seconddir) if ".tmp." not in n]
                except OSError:
                    continue
                rng.shuffle(names)
                for name in names:
                    picked.append(os.path.join(seconddir, name))
                    if len(picked) >= count:
                        return picked
        return picked

    def verify(self, sample=None, seed=0):
        """Re-hash stored objects and report mismatches.

        ``sample`` limits the check to a pseudo-random subset (by count),
        selected without enumerating the whole store.  Returns a dict with
        ``checked``, ``ok``, ``bad`` (list of digests) and ``hardlinks`` (the
        number of objects sharing an inode with something else, i.e. evidence
        the store is not holding private copies).
        """
        if sample:
            paths = self.sample_objects(sample, seed=seed)
        else:
            paths = list(self.iterobjects())
        report = {"checked": 0, "ok": 0, "bad": [], "hardlinks": 0, "bytes": 0}
        for path in paths:
            expect = os.path.basename(path)
            stat = os.stat(path)
            if stat.st_nlink > 1:
                report["hardlinks"] += 1
            got, size = self.hashfile(path)
            report["checked"] += 1
            report["bytes"] += size
            if got == expect:
                report["ok"] += 1
            else:
                report["bad"].append({"object": expect, "actual": got})
        return report

    def usage(self):
        """Return ``{objects, apparent_bytes, exclusive_bytes}``.

        ``exclusive_bytes`` counts only objects whose inode is not shared, so a
        correctly hardlinked store reports ~0 exclusive bytes.
        """
        objects = apparent = exclusive = 0
        for path in self.iterobjects():
            stat = os.stat(path, follow_symlinks=False)
            objects += 1
            apparent += stat.st_size
            if stat.st_nlink <= 1:
                exclusive += stat.st_size
        return {
            "objects": objects,
            "apparent_bytes": apparent,
            "exclusive_bytes": exclusive,
        }

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            if getattr(self._local, "pending", 0):
                try:
                    conn.commit()
                except sqlite3.Error:
                    pass
            conn.close()
            self._local.conn = None
            self._local.pending = 0


def _copyfile(src, dst):
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(_READ_CHUNK)
            if not chunk:
                break
            fout.write(chunk)
