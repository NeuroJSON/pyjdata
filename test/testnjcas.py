"""Tests for jdata.njcas -- the content-addressed store.

Correctness properties under test:
  * object names really are the SHA-256 of the file content
  * the git-annex key memo returns the same digest without re-reading
  * materialised objects share an inode with the source (no extra space)
  * verify() detects corruption
Performance property under test:
  * a warm (memoised) pass does no I/O on the payload, so re-running the
    pipeline over unchanged data is effectively free
"""

import os
import sys
import time
import shutil
import hashlib
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jdata.njcas import CAS, annex_key, annex_key_info, cas_url


class TestAnnexKeyParsing(unittest.TestCase):
    def test_key_from_symlink_target(self):
        root = tempfile.mkdtemp()
        try:
            key = "MD5E-s10132922--1bb65889704ae56474132752d155b88e.nii.gz"
            objdir = os.path.join(root, ".git", "annex", "objects", "Qv", "JM", key)
            os.makedirs(objdir)
            payload = os.path.join(objdir, key)
            with open(payload, "wb") as fid:
                fid.write(b"x")
            link = os.path.join(root, "sub-01_T1w.nii.gz")
            os.symlink(os.path.relpath(payload, root), link)
            self.assertEqual(annex_key(link), key)
        finally:
            shutil.rmtree(root)

    def test_non_annex_symlink_has_no_key(self):
        root = tempfile.mkdtemp()
        try:
            target = os.path.join(root, "real.txt")
            open(target, "w").close()
            link = os.path.join(root, "link.txt")
            os.symlink(target, link)
            self.assertIsNone(annex_key(link))
            self.assertIsNone(annex_key(target))
        finally:
            shutil.rmtree(root)

    def test_key_info_fields(self):
        info = annex_key_info("MD5E-s5663237--4608ffbd6b78ce3a325eb338fa556589.nii.gz")
        self.assertEqual(info["backend"], "MD5E")
        self.assertEqual(info["size"], 5663237)
        self.assertEqual(info["hash"], "4608ffbd6b78ce3a325eb338fa556589")
        self.assertEqual(info["ext"], ".nii.gz")

    def test_key_info_without_size(self):
        info = annex_key_info("SHA256--" + "a" * 64)
        self.assertEqual(info["backend"], "SHA256")
        self.assertIsNone(info["size"])

    def test_key_info_rejects_garbage(self):
        self.assertIsNone(annex_key_info("not-a-key"))
        self.assertIsNone(annex_key_info(None))


class TestCasUrl(unittest.TestCase):
    def test_hash_is_present_and_file_is_last(self):
        url = cas_url("a" * 64, size=42, db="db", doc="ds1", file="sub-01/x.nii.gz")
        self.assertIn("hash=sha256:" + "a" * 64, url)
        self.assertIn("size=42", url)
        # the CouchDB links view regex anchors its filename match at the end of
        # the string, so file= has to stay the final parameter
        self.assertTrue(url.endswith("file=sub-01/x.nii.gz"))

    def test_deterministic_parameter_order(self):
        args = dict(size=1, db="d", doc="s", file="f")
        self.assertEqual(cas_url("b" * 64, **args), cas_url("b" * 64, **args))

    def test_special_characters_are_escaped(self):
        url = cas_url("c" * 64, db="a b", doc="x&y", file="p/q r.nii")
        self.assertIn("db=a%20b", url)
        self.assertIn("doc=x%26y", url)
        self.assertIn("file=p/q%20r.nii", url)


class TestCasStore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.casroot = os.path.join(self.root, "cas")
        self.cas = CAS(self.casroot, commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, data):
        path = os.path.join(self.root, name)
        with open(path, "wb") as fid:
            fid.write(data)
        return path

    def test_digest_matches_hashlib(self):
        data = os.urandom(200000)
        path = self._write("a.bin", data)
        digest, size = self.cas.put(path)
        self.assertEqual(digest, hashlib.sha256(data).hexdigest())
        self.assertEqual(size, len(data))

    def test_object_is_hardlink_not_copy(self):
        path = self._write("b.bin", b"hello world" * 1000)
        digest, _ = self.cas.put(path)
        stored = self.cas.objpath(digest)
        self.assertTrue(os.path.exists(stored))
        self.assertEqual(os.stat(stored).st_ino, os.stat(path).st_ino)
        self.assertGreater(os.stat(stored).st_nlink, 1)
        # no extra space is consumed by a hardlinked store
        self.assertEqual(self.cas.usage()["exclusive_bytes"], 0)

    def test_identical_content_deduplicates(self):
        one = self._write("c1.bin", b"same bytes")
        two = self._write("c2.bin", b"same bytes")
        first, _ = self.cas.put(one)
        second, _ = self.cas.put(two)
        self.assertEqual(first, second)
        self.assertEqual(self.cas.usage()["objects"], 1)

    def test_put_is_idempotent(self):
        path = self._write("d.bin", b"abc" * 500)
        self.cas.put(path)
        before = self.cas.stats["linked"]
        self.cas.put(path)
        self.assertEqual(self.cas.stats["linked"], before)
        self.assertGreaterEqual(self.cas.stats["already"], 1)

    def test_memo_avoids_rehashing_annex_content(self):
        key = "MD5E-s24--" + "0" * 32 + ".bin"
        path = self._write("e.bin", b"annexed payload bytes!!!")
        first = self.cas.digest(path, key=key)
        hashed_before = self.cas.stats["hashed"]
        second = self.cas.digest(path, key=key)
        self.assertEqual(first, second)
        # the second call must not touch the payload at all
        self.assertEqual(self.cas.stats["hashed"], hashed_before)
        self.assertGreaterEqual(self.cas.stats["memo_hits"], 1)

    def test_mode_none_records_hash_without_materialising(self):
        cas = CAS(os.path.join(self.root, "cas2"), mode="none")
        path = self._write("f.bin", b"payload")
        digest, _ = cas.put(path)
        self.assertFalse(cas.has(digest))
        cas.close()

    def test_mode_copy_materialises_independent_bytes(self):
        cas = CAS(os.path.join(self.root, "cas3"), mode="copy")
        path = self._write("g.bin", b"copied payload")
        digest, _ = cas.put(path)
        stored = cas.objpath(digest)
        self.assertNotEqual(os.stat(stored).st_ino, os.stat(path).st_ino)
        with open(stored, "rb") as fid:
            self.assertEqual(fid.read(), b"copied payload")
        cas.close()

    def test_verify_reports_clean_store(self):
        for i in range(5):
            self.cas.put(self._write("h%d.bin" % i, os.urandom(1000)))
        report = self.cas.verify()
        self.assertEqual(report["checked"], 5)
        self.assertEqual(report["ok"], 5)
        self.assertEqual(report["bad"], [])
        self.assertEqual(report["hardlinks"], 5)

    def test_verify_detects_corruption(self):
        digest, _ = self.cas.put(self._write("i.bin", b"original"))
        stored = self.cas.objpath(digest)
        os.chmod(stored, 0o644)
        os.unlink(stored)
        with open(stored, "wb") as fid:
            fid.write(b"tampered")
        report = self.cas.verify()
        self.assertEqual(report["ok"], 0)
        self.assertEqual(len(report["bad"]), 1)
        self.assertEqual(report["bad"][0]["object"], digest)


def _machine_is_busy():
    """True when a throughput floor would measure contention, not regression.

    These assertions exist to catch a real slowdown, and an absolute MB/s floor
    cannot distinguish one from a machine that is simply saturated -- which is
    exactly what a bulk conversion does to the disks it is reading.
    """
    try:
        return os.getloadavg()[0] > max(2.0, (os.cpu_count() or 1) * 0.5)
    except (OSError, AttributeError):
        return False


class TestCasPerformance(unittest.TestCase):
    """Performance characteristics the bulk conversion depends on."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cas = CAS(os.path.join(self.root, "cas"))

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    @unittest.skipIf(_machine_is_busy(), "machine is loaded; throughput floor is meaningless")
    def test_cold_hash_throughput(self):
        # 32 MiB of incompressible data; asserts a floor far below what any
        # sane disk delivers, so this fails only on a real regression
        data = os.urandom(32 << 20)
        path = os.path.join(self.root, "big.bin")
        with open(path, "wb") as fid:
            fid.write(data)
        start = time.time()
        self.cas.put(path, key="MD5E-s%d--%s.bin" % (len(data), "1" * 32))
        elapsed = max(time.time() - start, 1e-6)
        rate = len(data) / elapsed / 1e6
        self.assertGreater(rate, 20.0, "hash throughput fell to %.1f MB/s" % rate)

    @unittest.skipIf(_machine_is_busy(), "machine is loaded; timing ratio is unreliable")
    def test_warm_pass_is_orders_of_magnitude_faster(self):
        data = os.urandom(16 << 20)
        path = os.path.join(self.root, "big2.bin")
        with open(path, "wb") as fid:
            fid.write(data)
        key = "MD5E-s%d--%s.bin" % (len(data), "2" * 32)
        start = time.time()
        self.cas.put(path, key=key)
        cold = max(time.time() - start, 1e-6)
        start = time.time()
        for _ in range(20):
            self.cas.put(path, key=key)
        warm = max((time.time() - start) / 20, 1e-9)
        self.assertLess(warm, cold / 10.0, "memo gave only %.1fx speedup" % (cold / warm))
        self.assertEqual(self.cas.stats["hashed"], 1)


if __name__ == "__main__":
    unittest.main()


def _hammer_memo(args):
    """Worker for the concurrency test: open the memo and put/get many keys."""
    casroot, worker, count = args
    cas = CAS(casroot, commit_every=8)
    errors = 0
    try:
        for i in range(count):
            key = "MD5E-s10--%032d.bin" % ((worker * count + i) % 97)
            cas.memo_put(key, "%064d" % i, 10)
            cas.memo_get(key)
        cas.flush()
        errors = cas.stats["memo_errors"]
    finally:
        cas.close()
    return errors


class TestMemoConcurrency(unittest.TestCase):
    """The memo must survive a wide process pool.

    Regression test: setting ``PRAGMA journal_mode`` on every connection needs
    an exclusive lock, so a pool of workers all opening the store at once
    serialised on that lock and eventually raised "database is locked" --
    failing whole datasets over a cache write.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.casroot = os.path.join(self.root, "cas")
        CAS(self.casroot).close()  # create the schema once, as the pipeline does

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_many_processes_share_the_memo_without_lock_errors(self):
        from concurrent.futures import ProcessPoolExecutor

        workers = 16
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_hammer_memo, [(self.casroot, w, 60) for w in range(workers)]))
        self.assertEqual(sum(results), 0, "memo reported lock/IO errors: %r" % results)
        cas = CAS(self.casroot)
        self.assertGreater(cas.memo_count(), 0)
        cas.close()

    def test_memo_failure_degrades_to_a_cache_miss(self):
        """A broken memo must never fail a conversion."""
        cas = CAS(self.casroot)
        payload = os.path.join(self.root, "x.bin")
        with open(payload, "wb") as fid:
            fid.write(b"content")
        cas.dbpath = os.path.join(self.root, "nonexistent-dir", "index.sqlite")
        cas._local = __import__("threading").local()
        digest, size = cas.digest(payload, key="MD5E-s7--" + "0" * 32 + ".bin")
        self.assertEqual(digest, hashlib.sha256(b"content").hexdigest())
        self.assertEqual(size, 7)
        self.assertGreater(cas.stats["memo_errors"], 0)
        cas.close()


class TestSampling(unittest.TestCase):
    """Sampling must not require enumerating the whole store."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)
        for i in range(40):
            path = os.path.join(self.root, "f%d.bin" % i)
            with open(path, "wb") as fid:
                fid.write(b"payload-%d" % i)
            self.cas.put(path)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_sample_returns_the_requested_count(self):
        picked = self.cas.sample_objects(10)
        self.assertEqual(len(picked), 10)
        self.assertEqual(len(set(picked)), 10)
        for path in picked:
            self.assertTrue(os.path.exists(path))

    def test_sample_is_capped_by_store_size(self):
        self.assertEqual(len(self.cas.sample_objects(1000)), 40)

    def test_sample_is_deterministic_for_a_given_seed(self):
        self.assertEqual(self.cas.sample_objects(8, seed=7), self.cas.sample_objects(8, seed=7))

    def test_different_seeds_generally_differ(self):
        self.assertNotEqual(self.cas.sample_objects(8, seed=1), self.cas.sample_objects(8, seed=99))

    def test_verify_with_sample_checks_only_the_sample(self):
        report = self.cas.verify(sample=6)
        self.assertEqual(report["checked"], 6)
        self.assertEqual(report["ok"], 6)
        self.assertEqual(report["bad"], [])


def _open_fresh(args):
    """Worker: construct a CAS on a brand-new store and use it."""
    casroot, index = args
    cas = CAS(casroot, commit_every=1)
    try:
        path = os.path.join(os.path.dirname(casroot), "p%d.bin" % index)
        with open(path, "wb") as fid:
            fid.write(b"payload %d" % index)
        digest, _size = cas.put(path)
        return digest is not None
    finally:
        cas.close()


class TestConcurrentStoreCreation(unittest.TestCase):
    """Creating the memo schema takes an exclusive lock.

    Regression test: a pool of workers all constructing a CAS on a brand-new
    store collided on that lock, and the exception escaped the constructor --
    failing the worker before it had converted anything.  Schema creation now
    degrades to "no memo" instead.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_many_workers_may_create_the_store_simultaneously(self):
        from concurrent.futures import ProcessPoolExecutor

        casroot = os.path.join(self.root, "cas")
        with ProcessPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(_open_fresh, [(casroot, i) for i in range(12)]))
        self.assertTrue(all(results))

    def test_store_on_a_read_only_parent_still_works_without_a_memo(self):
        casroot = os.path.join(self.root, "cas2")
        cas = CAS(casroot)
        cas.close()
        # make the memo unusable, then construct again
        os.chmod(os.path.join(casroot, "index.sqlite"), 0o000)
        try:
            cas = CAS(casroot)
            payload = os.path.join(self.root, "q.bin")
            with open(payload, "wb") as fid:
                fid.write(b"data")
            digest, size = cas.put(payload)
            self.assertEqual(size, 4)
            self.assertEqual(digest, hashlib.sha256(b"data").hexdigest())
            cas.close()
        finally:
            os.chmod(os.path.join(casroot, "index.sqlite"), 0o644)


class TestAnnexHashMode(unittest.TestCase):
    """Taking the content hash from the git-annex key costs no payload I/O.

    An MD5E key states both the content hash and the exact size, and hardlinking
    is a metadata operation, so a whole conversion pass can run without reading
    a single payload.  On a multi-terabyte mirror that read *is* the runtime.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.payload = b"annexed payload contents"
        self.md5 = hashlib.md5(self.payload).hexdigest()
        key = "MD5E-s%d--%s.bin" % (len(self.payload), self.md5)
        objdir = os.path.join(self.root, "ds", ".git", "annex", "objects", "aa", "bb", key)
        os.makedirs(objdir)
        target = os.path.join(objdir, key)
        with open(target, "wb") as fid:
            fid.write(self.payload)
        self.link = os.path.join(self.root, "ds", "file.bin")
        os.symlink(os.path.relpath(target, os.path.dirname(self.link)), self.link)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_digest_comes_from_the_key_and_is_correct(self):
        cas = CAS(os.path.join(self.root, "cas"), algo="md5", annex_hash=True)
        digest, size = cas.digest(self.link)
        self.assertEqual(digest, self.md5)
        self.assertEqual(size, len(self.payload))
        self.assertEqual(cas.stats["hashed"], 0)  # payload never read
        self.assertEqual(cas.stats["from_annex"], 1)
        cas.close()

    def test_object_is_still_hardlinked(self):
        cas = CAS(os.path.join(self.root, "cas2"), algo="md5", annex_hash=True)
        digest, _size = cas.put(self.link)
        stored = cas.objpath(digest)
        self.assertGreater(os.stat(stored).st_nlink, 1)
        self.assertEqual(cas.stats["bytes_hashed"], 0)
        cas.close()

    def test_algorithm_mismatch_falls_back_to_hashing(self):
        """A sha256 store must not accept an MD5E key's hash."""
        cas = CAS(os.path.join(self.root, "cas3"), algo="sha256", annex_hash=True)
        digest, _size = cas.digest(self.link)
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(cas.stats["from_annex"], 0)
        self.assertEqual(cas.stats["hashed"], 1)
        cas.close()

    def test_malformed_key_falls_back_to_hashing(self):
        cas = CAS(os.path.join(self.root, "cas4"), algo="md5", annex_hash=True)
        for bad in ("MD5E-s5--nothex.bin", "MD5E--%s.bin" % self.md5, "URL--http://x"):
            self.assertIsNone(cas.annex_digest(bad))
        cas.close()

    def test_disabled_by_default(self):
        cas = CAS(os.path.join(self.root, "cas5"), algo="md5")
        cas.digest(self.link)
        self.assertEqual(cas.stats["from_annex"], 0)
        cas.close()


class TestAnyAnnexBackend(unittest.TestCase):
    """The store must reuse whatever hash a git-annex key already carries.

    Regression test: the key's digest was only accepted when its backend matched
    the store's own algorithm, so a collection mixing MD5E and SHA256E (OpenNeuro
    does) had every file under the other backend read from disk to recompute a
    hash that was already sitting in its symlink target.  On one dataset that was
    40.65s for 1500 files instead of 0.51s.
    """

    ALGOS = {"MD5E": "md5", "SHA256E": "sha256", "SHA1E": "sha1"}

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _annexed(self, backend, payload, ext=".nii.gz"):
        digest = hashlib.new(self.ALGOS[backend], payload).hexdigest()
        key = "%s-s%d--%s%s" % (backend, len(payload), digest, ext)
        objdir = os.path.join(self.root, ".git", "annex", "objects", "aa", "bb", key)
        os.makedirs(objdir, exist_ok=True)
        target = os.path.join(objdir, key)
        with open(target, "wb") as fid:
            fid.write(payload)
        link = os.path.join(self.root, "f_%s%s" % (backend, ext))
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
        return link, digest

    def test_every_supported_backend_is_reused_without_reading(self):
        cas = CAS(os.path.join(self.root, "store"), algo="md5", annex_hash=True)
        try:
            for backend in ("MD5E", "SHA256E", "SHA1E"):
                link, digest = self._annexed(backend, b"payload for " + backend.encode())
                info = cas.identify(link, store=True)
                self.assertEqual(info["digest"], digest, backend)
                self.assertEqual(info["algo"], self.ALGOS[backend], backend)
                self.assertTrue(cas.has(info["digest"]), backend)
            self.assertEqual(cas.stats["hashed"], 0)
            self.assertEqual(cas.stats["from_annex"], 3)
        finally:
            cas.close()

    def test_object_is_materialised_by_hardlink(self):
        cas = CAS(os.path.join(self.root, "store2"), algo="md5", annex_hash=True)
        try:
            link, digest = self._annexed("SHA256E", b"a payload to hardlink")
            cas.identify(link, store=True)
            self.assertGreater(os.stat(cas.objpath(digest)).st_nlink, 1)
            self.assertEqual(cas.stats["bytes_hashed"], 0)
        finally:
            cas.close()

    def test_unhashed_backend_falls_back_to_reading(self):
        """A URL-backend key carries no content hash, so the payload is read."""
        cas = CAS(os.path.join(self.root, "store3"), algo="sha256", annex_hash=True)
        try:
            payload = b"content behind a URL key"
            key = "URL--http%3A%2F%2Fexample.invalid%2Ff.nii.gz"
            objdir = os.path.join(self.root, ".git", "annex", "objects", "cc", "dd", key)
            os.makedirs(objdir)
            target = os.path.join(objdir, key)
            with open(target, "wb") as fid:
                fid.write(payload)
            link = os.path.join(self.root, "url.nii.gz")
            os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
            info = cas.identify(link, store=True)
            self.assertEqual(info["algo"], "sha256")
            self.assertEqual(info["digest"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(cas.stats["hashed"], 1)
        finally:
            cas.close()

    def test_identify_without_storing_does_not_materialise(self):
        cas = CAS(os.path.join(self.root, "store4"), algo="md5", annex_hash=True)
        try:
            link, digest = self._annexed("SHA256E", b"not to be materialised")
            info = cas.identify(link, store=False)
            self.assertEqual(info["digest"], digest)
            self.assertFalse(cas.has(digest))
        finally:
            cas.close()

    def test_digest_still_returns_a_pair_for_its_own_algorithm(self):
        """digest() keeps its two-value contract for existing callers."""
        cas = CAS(os.path.join(self.root, "store5"), algo="md5", annex_hash=True)
        try:
            link, digest = self._annexed("MD5E", b"two value contract")
            self.assertEqual(cas.digest(link), (digest, len(b"two value contract")))
        finally:
            cas.close()
