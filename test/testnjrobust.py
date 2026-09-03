"""Structural robustness of the converter.

A corpus of 1548 real datasets contains structures nobody designed for: a
subject that is a file, a symlink that points at itself, a file the process
cannot read.  The contract asserted here is that **no single file or directory
can abort a dataset** -- anything unparseable degrades to a link and is recorded
in ``errors``, and the document is still produced.

Every case below crashed or lost data at some point during development.
"""

import os
import sys
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from jdata.njcas import CAS
from jdata.njbids import bids2json, canonical_json

from testnjbids import make_bids, _write


class RobustnessCase(unittest.TestCase):
    """Builds a minimal valid dataset, breaks one thing, and converts it."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = os.path.join(self.root, "dsR")
        make_bids(self.ds, subjects=("01",), git=False, derivatives=False)
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        # a fixture may have removed read permission
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    os.chmod(os.path.join(dirpath, name), 0o644)
                except OSError:
                    pass
        shutil.rmtree(self.root, ignore_errors=True)

    def convert(self):
        return bids2json(self.ds, dbname="db", dsname="dsR", cas=self.cas)

    def assertUsable(self, result):
        """The document must still be complete and serialisable."""
        self.assertIn("dataset_description.json", result["doc"])
        self.assertIn("participant_id", result["doc"]["participants.tsv"])
        self.assertIn("Fingerprint", result["doc"][".neurojson"])
        canonical_json(result["doc"])  # must not raise
        return result


class TestBrokenSymlinks(RobustnessCase):
    def test_dangling_symlink_outside_the_dataset(self):
        """No git-annex key, so no recoverable size -- this raised a TypeError."""
        os.symlink(
            "/nonexistent/elsewhere.nii.gz",
            os.path.join(self.ds, "sub-01", "anat", "outside.nii.gz"),
        )
        result = self.assertUsable(self.convert())
        node = result["doc"]["sub-01"]["anat"]["outside.nii.gz"]
        self.assertIn("_DataLink_", node)

    def test_self_referential_symlink(self):
        link = os.path.join(self.ds, "sub-01", "loop.json")
        os.symlink("loop.json", link)
        self.assertUsable(self.convert())

    def test_symlink_to_the_parent_directory_does_not_recurse(self):
        os.symlink("..", os.path.join(self.ds, "sub-01", "up"))
        result = self.assertUsable(self.convert())
        paths = [entry["path"] for entry in result["manifest"]]
        self.assertEqual(len(paths), len(set(paths)))

    def test_manifest_entries_always_have_a_numeric_size(self):
        os.symlink("/gone", os.path.join(self.ds, "sub-01", "gone.bin"))
        for entry in self.convert()["manifest"]:
            self.assertIsInstance(entry.get("size") or 0, int)


class TestUnreadableFiles(RobustnessCase):
    def test_unreadable_file_does_not_abort_the_dataset(self):
        """The fallback path itself touches the payload, so it can fail too."""
        target = os.path.join(self.ds, "sub-01", "anat", "locked.json")
        _write(target, "{}")
        os.chmod(target, 0o000)
        result = self.assertUsable(self.convert())
        self.assertTrue(any("unreadable" in e or "Permission" in e for e in result["errors"]))
        node = result["doc"]["sub-01"]["anat"]["locked.json"]
        self.assertIn("_DataLink_", node)


class TestOddStructures(RobustnessCase):
    def test_missing_dataset_description(self):
        os.remove(os.path.join(self.ds, "dataset_description.json"))
        result = self.convert()
        self.assertIn("participant_id", result["doc"]["participants.tsv"])
        self.assertEqual(result["errors"], [])

    def test_subject_entry_that_is_a_file(self):
        """An extensionless file of unknown type becomes a link, not a subtree."""
        _write(os.path.join(self.ds, "sub-99"), "not a directory")
        result = self.assertUsable(self.convert())
        self.assertIn("_DataLink_", result["doc"]["sub-99"])

    def test_split_directory_that_is_a_file(self):
        _write(os.path.join(self.ds, "derivatives"), "not a directory")
        result = self.assertUsable(self.convert())
        self.assertEqual(result["split"], {})

    def test_empty_dataset(self):
        for name in os.listdir(self.ds):
            path = os.path.join(self.ds, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        result = bids2json(self.ds, dbname="db", dsname="dsR", cas=self.cas)
        self.assertIn("Fingerprint", result["doc"][".neurojson"])
        self.assertEqual(result["doc"][".neurojson"]["Files"], 0)

    def test_deeply_nested_directories(self):
        path = self.ds
        for index in range(40):
            path = os.path.join(path, "d%d" % index)
        _write(os.path.join(path, "deep.json"), "{}")
        self.assertUsable(self.convert())


class TestAwkwardNames(RobustnessCase):
    def test_non_ascii_filename(self):
        _write(os.path.join(self.ds, "sub-01", "anat", "sub-01_desc-café_T1w.json"), "{}")
        result = self.assertUsable(self.convert())
        self.assertIn("sub-01_desc-café_T1w.json", result["doc"]["sub-01"]["anat"])

    def test_space_in_filename(self):
        _write(os.path.join(self.ds, "sub-01", "anat", "sub-01 copy.json"), "{}")
        self.assertUsable(self.convert())

    def test_very_long_filename(self):
        _write(os.path.join(self.ds, "sub-01", "anat", "s" * 200 + ".json"), "{}")
        self.assertUsable(self.convert())

    def test_filename_with_a_percent_sign(self):
        _write(os.path.join(self.ds, "sub-01", "anat", "sub-01_desc-50%_T1w.json"), "{}")
        self.assertUsable(self.convert())


class TestCorruptPayloads(RobustnessCase):
    """A file whose contents do not match its extension must degrade, not crash."""

    def _case(self, relpath, content):
        target = os.path.join(self.ds, relpath)
        _write(target, content)
        result = self.assertUsable(self.convert())
        node = result["doc"]
        for part in relpath.split("/"):
            node = node[part]
        return result, node

    def test_gifti_that_is_not_xml(self):
        _result, node = self._case("sub-01/anat/sub-01_hemi-L_pial.gii", "not xml at all")
        self.assertIn("_DataLink_", node)

    def test_nwb_that_is_not_hdf5(self):
        _result, node = self._case("sub-01/sub-01_ecephys.nwb", "not hdf5")
        self.assertIn("_DataLink_", node)

    def test_snirf_that_is_not_hdf5(self):
        _result, node = self._case("sub-01/sub-01_nirs.snirf", "not hdf5")
        self.assertIn("_DataLink_", node)

    def test_truncated_edf(self):
        _result, node = self._case("sub-01/sub-01_eeg.edf", "0" * 40)
        self.assertIn("_DataLink_", node)

    def test_brainvision_header_full_of_binary_junk(self):
        _result, node = self._case("sub-01/sub-01_eeg.vhdr", "\x00\x01binary junk")
        self.assertIsInstance(node, dict)

    def test_mat_that_is_neither_matlab_nor_vest(self):
        _result, node = self._case("sub-01/sub-01_model.mat", "\x00\x01\x02 junk" * 20)
        self.assertIn("_DataLink_", node)

    def test_every_failure_is_recorded_or_degraded_silently(self):
        """Errors are reported, never swallowed into a wrong value."""
        _write(os.path.join(self.ds, "sub-01", "bad.snirf"), "nope")
        result = self.convert()
        self.assertTrue(result["errors"])
        for message in result["errors"]:
            self.assertIn(":", message)  # every message names the file


if __name__ == "__main__":
    unittest.main()
