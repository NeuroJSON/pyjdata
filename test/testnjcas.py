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


class TestCasPerformance(unittest.TestCase):
    """Performance characteristics the bulk conversion depends on."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cas = CAS(os.path.join(self.root, "cas"))

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

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
