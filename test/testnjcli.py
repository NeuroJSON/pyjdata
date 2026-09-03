"""Tests for jdata.njcli output-layout helpers."""

import os
import sys
import json
import shutil
import tempfile
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
        self.vdir = os.path.join(self.root, "ds000001", "1.0.0")
        os.makedirs(self.vdir)
        for name in ("doc.json", "meta.json", "datacite.json", "derivatives.json"):
            with open(os.path.join(self.vdir, name), "w") as fid:
                fid.write("{}")
        with open(os.path.join(self.vdir, "manifest.tsv"), "w") as fid:
            fid.write("%s\t10\ta/b.nii.gz\n" % ("a" * 64))
        os.symlink("1.0.0", os.path.join(self.root, "ds000001", "latest"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_datacite_record_is_not_treated_as_a_document(self):
        items = list(_iter_published(self.root))
        self.assertEqual(len(items), 1)
        _ds, _version, _docpath, splits = items[0]
        self.assertNotIn("datacite", splits)

    def test_known_split_directory_is_published(self):
        _ds, _version, _docpath, splits = list(_iter_published(self.root))[0]
        self.assertIn("derivatives", splits)

    def test_unknown_json_is_ignored(self):
        with open(os.path.join(self.vdir, "scratch.json"), "w") as fid:
            fid.write("{}")
        _ds, _version, _docpath, splits = list(_iter_published(self.root))[0]
        self.assertNotIn("scratch", splits)

    def test_version_comes_from_the_latest_symlink(self):
        _ds, version, docpath, _splits = list(_iter_published(self.root))[0]
        self.assertEqual(version, "1.0.0")
        self.assertTrue(docpath.endswith(os.path.join("1.0.0", "doc.json")))

    def test_dataset_without_a_latest_symlink_is_skipped(self):
        other = os.path.join(self.root, "ds999999", "1.0.0")
        os.makedirs(other)
        with open(os.path.join(other, "doc.json"), "w") as fid:
            fid.write("{}")
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
            fid.write("None\tNone\tunfetched.nii.gz\n")
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
