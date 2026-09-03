"""Tests for jdata.njcli output-layout helpers."""

import os
import sys
import json
import shutil
import tempfile
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jdata.njcli import _iter_published, _read_manifest, _iter_links, NON_DOCUMENT_FILES


class TestIterPublished(unittest.TestCase):
    """Only real documents may be published.

    Regression test: the split-document test was "any .json that is not
    doc.json", so the DataCite record written beside a document was picked up as
    a derivatives document and pushed to the derivatives database.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.vdir = os.path.join(self.root, "ds000001")
        os.makedirs(self.vdir)
        for name in ("doc.json", "datacite.json", "derivatives.json"):
            with open(os.path.join(self.vdir, name), "w") as fid:
                fid.write("{}")
        with open(os.path.join(self.vdir, "meta.json"), "w") as fid:
            json.dump({"label": "1.0.0"}, fid)
        with open(os.path.join(self.vdir, "manifest.tsv"), "w") as fid:
            fid.write("md5:%s\t10\ta/b.nii.gz\n" % ("a" * 32))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_datacite_record_is_not_treated_as_a_document(self):
        items = list(_iter_published(self.root))
        self.assertEqual(len(items), 1)
        _ds, _label, _docpath, splits = items[0]
        self.assertNotIn("datacite", splits)

    def test_known_split_directory_is_published(self):
        _ds, _label, _docpath, splits = list(_iter_published(self.root))[0]
        self.assertIn("derivatives", splits)

    def test_unknown_json_is_ignored(self):
        with open(os.path.join(self.vdir, "scratch.json"), "w") as fid:
            fid.write("{}")
        _ds, _label, _docpath, splits = list(_iter_published(self.root))[0]
        self.assertNotIn("scratch", splits)

    def test_label_comes_from_the_metadata_file(self):
        _ds, label, docpath, _splits = list(_iter_published(self.root))[0]
        self.assertEqual(label, "1.0.0")
        self.assertTrue(docpath.endswith(os.path.join("ds000001", "doc.json")))

    def test_directory_without_a_document_is_skipped(self):
        os.makedirs(os.path.join(self.root, "ds999999"))
        names = [item[0] for item in _iter_published(self.root)]
        self.assertEqual(names, ["ds000001"])

    def test_non_document_files_are_declared(self):
        for name in ("doc.json", "meta.json", "datacite.json"):
            self.assertIn(name, NON_DOCUMENT_FILES)


class TestManifestReader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_reads_three_column_rows_and_skips_the_payload_line(self):
        path = os.path.join(self.root, "manifest.tsv")
        with open(path, "w") as fid:
            fid.write(
                "%s\t12\ta.nii.gz\n%s\t34\tb.tsv\npayload\t%s\n" % ("a" * 64, "b" * 64, "c" * 64)
            )
        entries = _read_manifest(path)
        self.assertEqual([e["path"] for e in entries], ["a.nii.gz", "b.tsv"])
        self.assertEqual(entries[0]["size"], 12)

    def test_unfetched_entries_have_no_hash(self):
        path = os.path.join(self.root, "m2.tsv")
        with open(path, "w") as fid:
            fid.write("md5:None\tNone\tunfetched.nii.gz\n")
        entry = _read_manifest(path)[0]
        self.assertIsNone(entry["sha256"])
        self.assertIsNone(entry["size"])

    def test_missing_file_yields_no_entries(self):
        self.assertEqual(_read_manifest(os.path.join(self.root, "nope.tsv")), [])


class TestLinkWalker(unittest.TestCase):
    def test_finds_hashes_and_reports_their_jsonpath(self):
        doc = {
            "sub-01": {
                "anat": {
                    "x.nii.gz": {
                        "NIFTIData": {"_DataLink_": "http://h/?hash=sha256:%s&size=1" % ("a" * 64)}
                    }
                }
            }
        }
        found = list(_iter_links(doc))
        self.assertEqual(len(found), 1)
        algo, digest, where = found[0]
        self.assertEqual(algo, "sha256")
        self.assertEqual(digest, "a" * 64)
        self.assertIn("sub-01", where)

    def test_escapes_dots_in_keys_so_the_path_is_a_valid_jsonpath(self):
        doc = {"a.b": {"_DataLink_": "http://h/?hash=md5:%s" % ("f" * 32)}}
        _algo, _digest, where = list(_iter_links(doc))[0]
        self.assertIn("a\\.b", where)

    def test_links_without_a_hash_are_ignored(self):
        doc = {"a": {"_DataLink_": "symlink:../elsewhere"}}
        self.assertEqual(list(_iter_links(doc)), [])

    def test_walks_into_lists(self):
        doc = {"a": [{"_DataLink_": "http://h/?hash=sha256:%s" % ("b" * 64)}]}
        self.assertEqual(len(list(_iter_links(doc))), 1)


class TestTrimLargestSubtree(unittest.TestCase):
    """Trimming happens at publish time, once the server has said no.

    A JSON byte count is the wrong thing to budget against, because CouchDB
    limits the *internal* size of a parsed document and the ratio to JSON
    depends entirely on content.  Measured against CouchDB 3.4.2 with an 8 MB
    limit, the largest JSON accepted was 4.19 MB for one big string, 7.23 MB for
    many short keys, and over 29 MB for a float array -- a sevenfold spread that
    no client-side estimate would predict.
    """

    def setUp(self):
        import shutil as _shutil

        self.root = tempfile.mkdtemp()
        self.docpath = os.path.join(self.root, "doc.json")
        from jdata.njcas import CAS

        self.cas = CAS(os.path.join(self.root, "store"), algo="md5", commit_every=1)
        self._shutil = _shutil

    def tearDown(self):
        self.cas.close()
        self._shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, doc):
        from jdata.njbids import canonical_json

        with open(self.docpath, "w", encoding="utf-8") as fid:
            fid.write(canonical_json(doc))

    def _read(self):
        with open(self.docpath, encoding="utf-8") as fid:
            return json.load(fid)

    def _doc(self):
        return {
            ".neurojson": {"Version": "1.0.0", "Fingerprint": "a" * 64},
            "dataset_description.json": {"Name": "probe"},
            "README": "keep me",
            "participants.tsv": {"participant_id": ["sub-01"], "age": [30]},
            "sub-01": {"anat": {"big.txt": "y" * 4000}},
            "sub-02": {"anat": {"mid.txt": "y" * 2000}},
            "sub-03": {"anat": {"small.txt": "y" * 500}},
        }

    def _trim(self):
        from jdata.njcli import _trim_largest_subtree

        return _trim_largest_subtree(self.docpath, self.cas, "db", "ds1")

    def test_sheds_the_largest_subtree_first(self):
        self._write(self._doc())
        shed = self._trim()
        self.assertEqual(shed["path"], "sub-01")
        self.assertLess(shed["now"], shed["was"] + 2000)

    def test_sheds_in_descending_size_order(self):
        self._write(self._doc())
        self.assertEqual(self._trim()["path"], "sub-01")
        self.assertEqual(self._trim()["path"], "sub-02")
        self.assertEqual(self._trim()["path"], "sub-03")
        self.assertIsNone(self._trim())

    def test_dataset_metadata_is_never_shed(self):
        self._write(self._doc())
        for _ in range(6):
            if self._trim() is None:
                break
        doc = self._read()
        self.assertEqual(doc["README"], "keep me")
        self.assertIn("Name", doc["dataset_description.json"])
        self.assertIn("participant_id", doc["participants.tsv"])
        self.assertIn("Fingerprint", doc[".neurojson"])

    def test_shed_subtree_is_replaced_by_a_resolvable_link(self):
        self._write(self._doc())
        self._trim()
        node = self._read()["sub-01"]
        self.assertEqual(list(node), ["_DataLink_"])
        digest = re.search(r"hash=md5:([0-9a-f]{32})", node["_DataLink_"]).group(1)
        self.assertTrue(os.path.exists(self.cas.objpath(digest)))

    def test_shed_subtree_content_is_recoverable_in_full(self):
        original = self._doc()
        self._write(original)
        self._trim()
        node = self._read()["sub-01"]
        digest = re.search(r"hash=md5:([0-9a-f]{32})", node["_DataLink_"]).group(1)
        with open(self.cas.objpath(digest), encoding="utf-8") as fid:
            self.assertEqual(json.load(fid), original["sub-01"])

    def test_document_on_disk_is_rewritten_to_match_what_is_published(self):
        self._write(self._doc())
        before = os.path.getsize(self.docpath)
        self._trim()
        self.assertLess(os.path.getsize(self.docpath), before)

    def test_already_linked_subtree_is_not_shed_again(self):
        doc = self._doc()
        doc["sub-01"] = {"_DataLink_": "http://h/?hash=md5:%s" % ("f" * 32)}
        self._write(doc)
        self.assertEqual(self._trim()["path"], "sub-02")

    def test_returns_none_when_only_protected_keys_remain(self):
        self._write(
            {
                ".neurojson": {"Version": "1"},
                "dataset_description.json": {"Name": "x"},
                "README": "y",
            }
        )
        self.assertIsNone(self._trim())
