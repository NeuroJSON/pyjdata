"""Tests for jdata.njbids -- BIDS dataset to version-invariant JSON digest.

The properties under test are the ones the versioning and DOI scheme rest on:

  * the document is byte-reproducible from the same input
  * the fingerprint changes when content changes, and does *not* change when
    only the download URL template changes
  * every file appears exactly once in the manifest
  * the file tree maps onto nested JSON keys the way the CouchDB views expect
  * an oversized document deterministically offloads until it fits
"""

import os
import sys
import json
import gzip
import shutil
import struct
import subprocess
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import collections

import numpy as np

from jdata.njcas import CAS
from jdata.njbids import (
    FileInfo,
    bids2json,
    strip_trailing_commas,
    canonical_json,
    dataset_version,
    fingerprint,
    fileext,
    _dehydrate,
)


def _walk_one(path, relpath):
    """Build the FileInfo the directory walk would have produced for one path."""
    is_link = os.path.islink(path)
    target = os.readlink(path) if is_link else None
    present = os.path.exists(path)
    size = os.path.getsize(path) if present else 0
    return FileInfo(path, relpath, is_link, present, size, target)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fid:
        fid.write(text)


def make_nifti(path, dims=(4, 4, 2), dtype=16, voxel=(2.0, 2.0, 3.0), gz=True):
    """Write a minimal but valid NIfTI-1 volume."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<h", hdr, 40, len(dims))
    for i, val in enumerate(dims):
        struct.pack_into("<h", hdr, 42 + 2 * i, val)
    struct.pack_into("<h", hdr, 70, dtype)  # float32
    struct.pack_into("<h", hdr, 72, 32)  # bitpix
    struct.pack_into("<f", hdr, 76, 1.0)  # pixdim[0]
    for i, val in enumerate(voxel):
        struct.pack_into("<f", hdr, 80 + 4 * i, val)
    struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)  # scl_slope
    hdr[344:348] = b"n+1\x00"
    body = bytes(hdr) + b"\x00" * 4 + np.zeros(int(np.prod(dims)), np.float32).tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if gz:
        with gzip.open(path, "wb") as fid:
            fid.write(body)
    else:
        with open(path, "wb") as fid:
            fid.write(body)
    return path


def make_bids(root, subjects=("01", "02"), git=False, derivatives=True):
    """Create a small but structurally complete BIDS dataset."""
    _write(
        os.path.join(root, "dataset_description.json"),
        json.dumps(
            {
                "Name": "Synthetic Test Dataset",
                "BIDSVersion": "1.8.0",
                "License": "CC0",
                "Authors": ["A Person", "B Person"],
                "DatasetDOI": "doi:10.18112/openneuro.dsTEST.v2.3.1",
            }
        ),
    )
    _write(os.path.join(root, "README"), "A synthetic dataset used by the test suite.\n")
    _write(os.path.join(root, "CHANGES"), "1.0.0 2024-01-01\n  - first\n")
    rows = ["participant_id\tage\tsex\thandedness"]
    for i, sub in enumerate(subjects):
        rows.append("sub-%s\t%d\t%s\tR" % (sub, 20 + i, "F" if i % 2 == 0 else "M"))
    _write(os.path.join(root, "participants.tsv"), "\n".join(rows) + "\n")
    _write(
        os.path.join(root, "participants.json"),
        json.dumps({"age": {"Units": "years"}, "sex": {"Description": "sex"}}),
    )
    for sub in subjects:
        base = os.path.join(root, "sub-%s" % sub)
        make_nifti(os.path.join(base, "anat", "sub-%s_T1w.nii.gz" % sub))
        _write(
            os.path.join(base, "anat", "sub-%s_T1w.json" % sub),
            json.dumps({"EchoTime": 0.003, "Manufacturer": "Siemens"}),
        )
        make_nifti(
            os.path.join(base, "func", "sub-%s_task-rest_bold.nii.gz" % sub),
            dims=(4, 4, 2, 5),
        )
        _write(
            os.path.join(base, "func", "sub-%s_task-rest_events.tsv" % sub),
            "onset\tduration\ttrial_type\n0.0\t1.0\tgo\n2.0\t1.0\tstop\n",
        )
        _write(os.path.join(base, "dwi", "sub-%s_dwi.bval" % sub), "0 1000 1000\n")
        _write(
            os.path.join(base, "dwi", "sub-%s_dwi.bvec" % sub),
            "0 1 0\n0 0 1\n0 0 0\n",
        )
    if derivatives:
        _write(
            os.path.join(root, "derivatives", "fmriprep", "dataset_description.json"),
            json.dumps({"Name": "fmriprep", "DatasetType": "derivative"}),
        )
        make_nifti(
            os.path.join(
                root, "derivatives", "fmriprep", "sub-01", "sub-01_desc-preproc_bold.nii.gz"
            )
        )
    _write(os.path.join(root, "sourcedata", "notes.txt"), "raw scanner notes\n")
    if git:
        env = dict(
            os.environ,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@t",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@t",
        )
        run = lambda *a: subprocess.run(
            ["git", "-C", root] + list(a), capture_output=True, env=env, check=True
        )
        run("init", "-q")
        run("add", "-A")
        run("commit", "-qm", "initial")
    return root


class TestFileExt(unittest.TestCase):
    def test_compound_extensions(self):
        self.assertEqual(fileext("a/b/x.nii.gz"), ".nii.gz")
        self.assertEqual(fileext("x.tsv.gz"), ".tsv.gz")
        self.assertEqual(fileext("x.nii"), ".nii")
        self.assertEqual(fileext("X.TSV"), ".tsv")

    def test_dotted_directory_does_not_confuse_extension(self):
        # the shell implementation used ${ff#*.} and mis-parsed this
        self.assertEqual(fileext("ses-1.5T/sub-01_T1w.nii.gz"), ".nii.gz")


class TestConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.ds = make_bids(os.path.join(cls.root, "dsTEST"), git=True)
        cls.cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        cls.result = bids2json(cls.ds, dbname="testdb", dsname="dsTEST", cas=cls.cas)

    @classmethod
    def tearDownClass(cls):
        cls.cas.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_no_errors(self):
        self.assertEqual(self.result["errors"], [])

    def test_top_level_keys_match_bids_filenames(self):
        keys = set(self.result["doc"])
        for expected in [
            "dataset_description.json",
            "README",
            "CHANGES",
            "participants.tsv",
            "participants.json",
            "sub-01",
            "sub-02",
            ".neurojson",
        ]:
            self.assertIn(expected, keys)

    def test_path_becomes_nested_keys(self):
        doc = self.result["doc"]
        self.assertIn("sub-01_T1w.nii.gz", doc["sub-01"]["anat"])
        self.assertIn("sub-01_task-rest_events.tsv", doc["sub-01"]["func"])

    def test_text_files_are_plain_strings(self):
        self.assertIsInstance(self.result["doc"]["README"], str)
        self.assertIn("synthetic", self.result["doc"]["README"])

    def test_json_sidecar_is_parsed_not_stringified(self):
        side = self.result["doc"]["sub-01"]["anat"]["sub-01_T1w.json"]
        self.assertEqual(side["EchoTime"], 0.003)

    def test_tsv_is_column_oriented(self):
        """The dbinfo/subjects views index participants.tsv by column arrays."""
        table = self.result["doc"]["participants.tsv"]
        self.assertEqual(table["participant_id"], ["sub-01", "sub-02"])
        self.assertEqual(list(table["age"]), [20, 21])
        self.assertEqual(table["sex"], ["F", "M"])

    def test_nifti_header_inlined_and_payload_linked(self):
        node = self.result["doc"]["sub-01"]["anat"]["sub-01_T1w.nii.gz"]
        self.assertIn("NIFTIHeader", node)
        self.assertIn("NIFTIData", node)
        hdr = node["NIFTIHeader"]
        self.assertEqual(list(hdr["Dim"]), [4, 4, 2])
        self.assertEqual([round(v, 3) for v in hdr["VoxelSize"]], [2.0, 2.0, 3.0])
        self.assertEqual(hdr["DataType"], "single")  # jdata uses MATLAB type names
        # the voxels themselves are a link, never inline data
        self.assertEqual(list(node["NIFTIData"]), ["_DataLink_"])
        self.assertIn("hash=sha256:", node["NIFTIData"]["_DataLink_"])

    def test_bval_bvec_are_numeric_arrays(self):
        bval = self.result["doc"]["sub-01"]["dwi"]["sub-01_dwi.bval"]
        self.assertEqual(list(np.asarray(bval).ravel()), [0.0, 1000.0, 1000.0])

    def test_derivatives_split_into_own_document(self):
        self.assertIn("derivatives", self.result["split"])
        deriv = self.result["split"]["derivatives"]
        self.assertIn("fmriprep", deriv)
        # and the main document keeps only a cross-reference
        self.assertEqual(list(self.result["doc"]["derivatives"]), ["_DataLink_"])
        self.assertIn("testdb_derivative", self.result["doc"]["derivatives"]["_DataLink_"])

    def test_sourcedata_is_link_only(self):
        node = self.result["doc"]["sourcedata"]["notes.txt"]
        self.assertEqual(list(node), ["_DataLink_"])

    def test_manifest_covers_every_file_exactly_once(self):
        paths = [entry["path"] for entry in self.result["manifest"]]
        self.assertEqual(len(paths), len(set(paths)))
        on_disk = set()
        for dirpath, dirnames, filenames in os.walk(self.ds):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), self.ds)
                on_disk.add(rel.replace(os.sep, "/"))
        # derivatives live in the split document, so exclude them here
        main = {p for p in on_disk if not p.startswith("derivatives/")}
        self.assertEqual(set(paths) & main, main)

    def test_manifest_totals_match_metadata_block(self):
        meta = self.result["doc"][".neurojson"]
        self.assertEqual(meta["Files"], len(self.result["manifest"]))
        self.assertEqual(meta["Bytes"], sum(e["size"] or 0 for e in self.result["manifest"]))

    def test_version_comes_from_dataset_doi_when_untagged(self):
        version = self.result["version"]
        self.assertEqual(version["Version"], "2.3.1")
        self.assertEqual(version["VersionSource"], "dataset_description.DatasetDOI")
        self.assertTrue(version["SourceCommit"])


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = make_bids(os.path.join(self.root, "dsD"), git=True)
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _convert(self, **kwargs):
        return bids2json(self.ds, dbname="db", dsname="dsD", cas=self.cas, **kwargs)

    def test_repeated_conversion_is_byte_identical(self):
        first = canonical_json(self._convert()["doc"])
        second = canonical_json(self._convert()["doc"])
        self.assertEqual(first, second)

    def test_fingerprint_is_stable(self):
        self.assertEqual(self._convert()["fingerprint"], self._convert()["fingerprint"])

    def test_fingerprint_survives_a_url_template_change(self):
        """Relocating the download endpoint must not invalidate a minted DOI."""
        one = self._convert(cas_url="https://a.example/get?x=1")
        two = self._convert(cas_url="https://totally-different.example/dl?y=2")
        self.assertEqual(one["fingerprint"], two["fingerprint"])
        # ...while the documents themselves genuinely differ
        self.assertNotEqual(canonical_json(one["doc"]), canonical_json(two["doc"]))

    def test_fingerprint_changes_when_content_changes(self):
        before = self._convert()["fingerprint"]
        _write(os.path.join(self.ds, "README"), "edited content\n")
        after = self._convert()["fingerprint"]
        self.assertNotEqual(before, after)

    def test_fingerprint_changes_when_a_payload_changes(self):
        before = self._convert()["fingerprint"]
        make_nifti(
            os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.nii.gz"),
            dims=(8, 8, 4),
        )
        self.assertNotEqual(before, self._convert()["fingerprint"])

    def test_canonical_json_is_sorted_and_compact(self):
        text = canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
        self.assertEqual(text, '{"a":{"c":3,"d":2},"b":1}')

    def test_canonical_json_replaces_non_finite_floats(self):
        text = canonical_json({"x": float("nan"), "y": [float("inf"), 1.5]})
        self.assertEqual(json.loads(text), {"x": None, "y": [None, 1.5]})

    def test_dehydrate_reduces_links_to_bare_hashes(self):
        node = {"a": {"_DataLink_": "https://h/x?hash=sha256:" + "f" * 64 + "&size=1"}}
        self.assertEqual(_dehydrate(node)["a"]["_DataLink_"], "sha256:" + "f" * 64)


class TestSizeBudget(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = make_bids(os.path.join(self.root, "dsB"), git=True, derivatives=False)
        # a large events table, well past any reasonable document budget
        rows = ["onset\tduration\ttrial_type\tresponse_time\tstim_file"]
        for i in range(20000):
            rows.append(
                "%f\t1.0\tcondition_%d\t%f\tstimuli/img_%05d.png" % (i * 0.5, i % 7, i * 0.001, i)
            )
        _write(
            os.path.join(self.ds, "sub-01", "func", "sub-01_task-rest_events.tsv"),
            "\n".join(rows) + "\n",
        )
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_document_is_kept_under_budget(self):
        budget = 60000
        result = bids2json(
            self.ds,
            dbname="db",
            dsname="dsB",
            cas=self.cas,
            max_doc=budget,
            max_tsv=1 << 30,
        )
        self.assertLessEqual(len(canonical_json(result["doc"])), budget)
        self.assertTrue(result["stats"]["offloaded"])
        offloaded = {item["path"] for item in result["stats"]["offloaded"]}
        self.assertIn("sub-01/func/sub-01_task-rest_events.tsv", offloaded)
        node = result["doc"]["sub-01"]["func"]["sub-01_task-rest_events.tsv"]
        self.assertEqual(list(node), ["_DataLink_"])

    def test_offload_choice_is_deterministic(self):
        args = dict(dbname="db", dsname="dsB", cas=self.cas, max_doc=60000, max_tsv=1 << 30)
        one = bids2json(self.ds, **args)
        two = bids2json(self.ds, **args)
        self.assertEqual(
            [i["path"] for i in one["stats"]["offloaded"]],
            [i["path"] for i in two["stats"]["offloaded"]],
        )
        self.assertEqual(one["fingerprint"], two["fingerprint"])

    def test_no_offload_when_document_already_fits(self):
        result = bids2json(self.ds, dbname="db", dsname="dsB", cas=self.cas, max_doc=50 << 20)
        self.assertEqual(result["stats"]["offloaded"], [])

    def test_participants_table_is_never_size_capped(self):
        """participants.tsv drives subject search, so it stays inline."""
        result = bids2json(self.ds, dbname="db", dsname="dsB", cas=self.cas, max_tsv=1)
        self.assertIn("participant_id", result["doc"]["participants.tsv"])


class TestVersionResolution(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _repo(self, name):
        path = os.path.join(self.root, name)
        make_bids(path, git=True, derivatives=False)
        return path

    def _tag(self, path, *tags):
        for tag in tags:
            subprocess.run(["git", "-C", path, "tag", tag], capture_output=True, check=True)

    def test_semver_tag_at_head_wins_over_doi(self):
        path = self._repo("a")
        self._tag(path, "1.0.0", "3.1.0", "2.0.0")
        info = dataset_version(path, {"DatasetDOI": "doi:10.18112/openneuro.x.v2.3.1"})
        self.assertEqual(info["Version"], "3.1.0")
        self.assertEqual(info["VersionSource"], "git-tag")

    def test_non_semver_tags_are_ignored(self):
        path = self._repo("b")
        self._tag(path, "00006", "57fecb0ccce88d000ac17538")
        info = dataset_version(path, {})
        self.assertEqual(info["VersionSource"], "git-commit")
        self.assertTrue(info["Version"].startswith("commit-"))

    def test_doi_used_when_no_semver_tag(self):
        path = self._repo("c")
        info = dataset_version(path, {"DatasetDOI": "10.18112/openneuro.ds1.v1.2.3"})
        self.assertEqual(info["Version"], "1.2.3")

    def test_commit_fallback_for_untagged_undoied_dataset(self):
        path = self._repo("d")
        info = dataset_version(path, {})
        self.assertEqual(info["VersionSource"], "git-commit")
        self.assertEqual(len(info["Version"]), len("commit-") + 8)

    def test_non_git_directory_is_handled(self):
        path = os.path.join(self.root, "plain")
        make_bids(path, git=False, derivatives=False)
        info = dataset_version(path, {})
        self.assertIsNone(info["SourceCommit"])
        self.assertEqual(info["Version"], "commit-unknown")


class TestFingerprintFunction(unittest.TestCase):
    def test_manifest_order_does_not_matter(self):
        manifest = [
            {"path": "b", "sha256": "1" * 64, "size": 2},
            {"path": "a", "sha256": "0" * 64, "size": 1},
        ]
        one, _ = fingerprint({"k": 1}, manifest)
        two, _ = fingerprint({"k": 1}, list(reversed(manifest)))
        self.assertEqual(one, two)

    def test_changing_a_hash_changes_the_fingerprint(self):
        base = [{"path": "a", "sha256": "0" * 64, "size": 1}]
        other = [{"path": "a", "sha256": "1" * 64, "size": 1}]
        self.assertNotEqual(fingerprint({}, base)[0], fingerprint({}, other)[0])

    def test_manifest_blob_is_tab_separated_and_sorted(self):
        manifest = [
            {"path": "z", "sha256": "1" * 64, "size": 2},
            {"path": "a", "sha256": "0" * 64, "size": 1},
        ]
        _digest, blob = fingerprint({}, manifest)
        lines = blob.strip().split("\n")
        self.assertTrue(lines[0].endswith("\ta"))
        self.assertTrue(lines[1].endswith("\tz"))
        self.assertTrue(lines[-1].startswith("payload\t"))


if __name__ == "__main__":
    unittest.main()


class TestMalformedInputs(unittest.TestCase):
    """Real corpora contain files that do not quite match their extension."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = make_bids(os.path.join(self.root, "dsM"), git=True, derivatives=False)
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _convert(self):
        return bids2json(self.ds, dbname="db", dsname="dsM", cas=self.cas)

    def test_json_sidecar_with_a_utf8_bom_is_parsed(self):
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.json")
        with open(target, "wb") as fid:
            fid.write(b'\xef\xbb\xbf{\r\n "EchoTime": 0.005\r\n}\r\n')
        result = self._convert()
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["doc"]["sub-01"]["anat"]["sub-01_T1w.json"]["EchoTime"], 0.005)

    def test_invalid_json_is_kept_as_text_not_dropped(self):
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.json")
        with open(target, "w") as fid:
            fid.write('# a comment line\n{"EchoTime": 1}\n')
        result = self._convert()
        node = result["doc"]["sub-01"]["anat"]["sub-01_T1w.json"]
        self.assertIsInstance(node, str)
        self.assertIn("EchoTime", node)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("invalid JSON kept as text", result["errors"][0])

    def test_tsv_with_latin1_units_column_is_parsed(self):
        target = os.path.join(self.ds, "sub-01", "func", "sub-01_task-rest_channels.tsv")
        with open(target, "wb") as fid:
            fid.write(b"name\tunits\nCh1\t\xb5V\nCh2\t\xb5V\n")
        result = self._convert()
        self.assertEqual(result["errors"], [])
        table = result["doc"]["sub-01"]["func"]["sub-01_task-rest_channels.tsv"]
        self.assertEqual(table["units"], ["µV", "µV"])

    def test_zero_byte_file_becomes_an_empty_object(self):
        target = os.path.join(self.ds, "sub-01", "anat", "empty.json")
        open(target, "w").close()
        result = self._convert()
        self.assertEqual(result["doc"]["sub-01"]["anat"]["empty.json"], {})

    def test_unparseable_nifti_falls_back_to_a_link(self):
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.nii.gz")
        with open(target, "wb") as fid:
            fid.write(b"not a nifti at all")
        result = self._convert()
        node = result["doc"]["sub-01"]["anat"]["sub-01_T1w.nii.gz"]
        self.assertEqual(list(node), ["_DataLink_"])
        self.assertTrue(any("nifti header" in e for e in result["errors"]))

    def test_a_single_bad_file_does_not_abort_the_dataset(self):
        with open(os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.nii.gz"), "wb") as fid:
            fid.write(b"garbage")
        with open(os.path.join(self.ds, "sub-02", "anat", "sub-02_T1w.json"), "w") as fid:
            fid.write("{not json")
        result = self._convert()
        # sub-02's other files still converted, and the document is complete
        self.assertIn("sub-02_T1w.nii.gz", result["doc"]["sub-02"]["anat"])
        self.assertIn("participant_id", result["doc"]["participants.tsv"])
        self.assertEqual(len(result["errors"]), 2)

    def test_tsv_containing_nul_bytes_is_parsed(self):
        """Python's csv raises on a stray NUL, and Postgres jsonb rejects it too."""
        target = os.path.join(self.ds, "sub-01", "func", "sub-01_task-rest_events.tsv")
        with open(target, "wb") as fid:
            fid.write(b"onset\tduration\ttrial_type\n0.0\t1.0\tgo\x00\n2.0\t1.0\tst\x00op\n")
        result = self._convert()
        self.assertEqual(result["errors"], [])
        table = result["doc"]["sub-01"]["func"]["sub-01_task-rest_events.tsv"]
        self.assertEqual(table["trial_type"], ["go", "stop"])
        self.assertNotIn("\x00", canonical_json(result["doc"]))

    def test_trailing_comma_sidecar_is_repaired_not_stringified(self):
        """Trailing commas are a common hand-editing artefact in BIDS sidecars."""
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.json")
        with open(target, "w") as fid:
            fid.write(
                '{\n "SamplingFrequency": 200.0,\n "Columns": [\n  "time",\n  "eeg",\n ]\n}\n'
            )
        result = self._convert()
        node = result["doc"]["sub-01"]["anat"]["sub-01_T1w.json"]
        self.assertIsInstance(node, dict)
        self.assertEqual(node["SamplingFrequency"], 200.0)
        self.assertEqual(node["Columns"], ["time", "eeg"])
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("repaired invalid JSON", result["errors"][0])

    def test_whitespace_only_json_is_treated_as_empty(self):
        """Several OpenNeuro datasets ship one-byte "\\n" sidecar placeholders."""
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_T1w.json")
        with open(target, "w") as fid:
            fid.write("\n")
        result = self._convert()
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["doc"]["sub-01"]["anat"]["sub-01_T1w.json"], {})


class TestBudgetTierTwo(unittest.TestCase):
    """A document can exceed the budget on links alone.

    Some datasets hold six figures of files.  At a couple of hundred bytes per
    ``_DataLink_`` that is past any document limit before a single byte of
    inline content is counted, so offloading leaf payloads cannot help and whole
    subtrees have to be shed instead.  What must never be shed is the
    dataset-level metadata that makes the document findable.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.ds = os.path.join(cls.root, "dsT2")
        make_bids(cls.ds, subjects=("01",), git=True, derivatives=False)
        # many link-only files across several subjects
        for sub in ("02", "03", "04"):
            for i in range(60):
                path = os.path.join(
                    cls.ds,
                    "sub-%s" % sub,
                    "func",
                    "sub-%s_task-rest_run-%03d_bold.nii.gz" % (sub, i),
                )
                make_nifti(path, dims=(2, 2, 2))
        cls.cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        cls.result = bids2json(cls.ds, dbname="db", dsname="dsT2", cas=cls.cas, max_doc=4000)

    @classmethod
    def tearDownClass(cls):
        cls.cas.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_document_fits_the_budget(self):
        self.assertLessEqual(len(canonical_json(self.result["doc"])), 4000)

    def test_subtrees_were_offloaded(self):
        kinds = {item["how"] for item in self.result["stats"]["offloaded"]}
        self.assertIn("subtree", kinds)

    def test_dataset_level_metadata_is_never_offloaded(self):
        doc = self.result["doc"]
        self.assertIsInstance(doc["dataset_description.json"], dict)
        self.assertIn("Name", doc["dataset_description.json"])
        self.assertIn("participant_id", doc["participants.tsv"])
        self.assertIsInstance(doc["README"], str)
        self.assertIn("Fingerprint", doc[".neurojson"])

    def test_offloaded_subtree_is_retrievable_from_the_store(self):
        offloaded = [i for i in self.result["stats"]["offloaded"] if i["how"] == "subtree"]
        self.assertTrue(offloaded)
        key = offloaded[0]["path"]
        node = self.result["doc"][key]
        self.assertEqual(list(node), ["_DataLink_"])
        match = re.search(r"hash=sha256:([0-9a-f]{64})", node["_DataLink_"])
        self.assertTrue(match)
        stored = self.cas.objpath(match.group(1))
        self.assertTrue(os.path.exists(stored))
        with open(stored, encoding="utf-8") as fid:
            recovered = json.load(fid)
        # the full subtree really is in there, not a truncation
        self.assertIsInstance(recovered, dict)
        self.assertTrue(recovered)

    def test_offload_selection_is_deterministic(self):
        again = bids2json(self.ds, dbname="db", dsname="dsT2", cas=self.cas, max_doc=4000)
        self.assertEqual(
            [i["path"] for i in again["stats"]["offloaded"]],
            [i["path"] for i in self.result["stats"]["offloaded"]],
        )
        self.assertEqual(again["fingerprint"], self.result["fingerprint"])


class TestSplitDocumentBudget(unittest.TestCase):
    """Split documents are published too, so they get the same budget."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.ds = os.path.join(cls.root, "dsSB")
        make_bids(cls.ds, subjects=("01",), git=True, derivatives=True)
        for pipeline in ("fmriprep", "freesurfer"):
            for i in range(50):
                make_nifti(
                    os.path.join(
                        cls.ds,
                        "derivatives",
                        pipeline,
                        "sub-01",
                        "sub-01_run-%03d_desc-preproc_bold.nii.gz" % i,
                    ),
                    dims=(2, 2, 2),
                )
        cls.cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        cls.result = bids2json(cls.ds, dbname="db", dsname="dsSB", cas=cls.cas, max_doc=3000)

    @classmethod
    def tearDownClass(cls):
        cls.cas.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_split_document_also_fits_the_budget(self):
        deriv = self.result["split"]["derivatives"]
        self.assertLessEqual(len(canonical_json(deriv)), 3000)

    def test_split_offloads_are_recorded_separately(self):
        self.assertIn("split_offloaded", self.result["stats"])
        self.assertIn("derivatives", self.result["stats"]["split_offloaded"])

    def test_offloaded_pipeline_is_a_resolvable_link(self):
        deriv = self.result["split"]["derivatives"]
        linked = [k for k, v in deriv.items() if isinstance(v, dict) and "_DataLink_" in v]
        self.assertTrue(linked)
        match = re.search(r"hash=sha256:([0-9a-f]{64})", deriv[linked[0]]["_DataLink_"])
        self.assertTrue(os.path.exists(self.cas.objpath(match.group(1))))


class TestFileLevelParallelism(unittest.TestCase):
    """Pre-hashing must change performance, never output."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = make_bids(os.path.join(self.root, "dsFP"), subjects=("01", "02"), git=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_output_is_identical_with_and_without_prehashing(self):
        serial = CAS(os.path.join(self.root, "cas1"), commit_every=1)
        parallel = CAS(os.path.join(self.root, "cas2"), commit_every=1)
        try:
            one = bids2json(self.ds, dbname="db", dsname="dsFP", cas=serial, hash_threads=1)
            two = bids2json(self.ds, dbname="db", dsname="dsFP", cas=parallel, hash_threads=8)
            self.assertEqual(canonical_json(one["doc"]), canonical_json(two["doc"]))
            self.assertEqual(one["fingerprint"], two["fingerprint"])
            self.assertEqual(
                [e["path"] for e in one["manifest"]], [e["path"] for e in two["manifest"]]
            )
            self.assertEqual(
                [e["sha256"] for e in one["manifest"]], [e["sha256"] for e in two["manifest"]]
            )
        finally:
            serial.close()
            parallel.close()

    def test_prehash_targets_only_annexed_payloads(self):
        """A non-annex tree has nothing to pre-hash, so the pass is a no-op."""
        from jdata.njbids import _prehash, _walk

        cas = CAS(os.path.join(self.root, "cas3"), commit_every=1)
        try:
            self.assertEqual(_prehash(_walk(self.ds), cas, 4), 0)
            self.assertEqual(cas.stats["hashed"], 0)
        finally:
            cas.close()

    def test_prehash_registers_annexed_files(self):
        from jdata.njbids import _prehash

        cas = CAS(os.path.join(self.root, "cas4"), commit_every=1)
        try:
            payload_dir = os.path.join(
                self.root, "annexlike", ".git", "annex", "objects", "aa", "bb"
            )
            files = []
            for i in range(6):
                key = "MD5E-s%d--%032d.bin" % (10 + i, i)
                objdir = os.path.join(payload_dir, key)
                os.makedirs(objdir, exist_ok=True)
                target = os.path.join(objdir, key)
                with open(target, "wb") as fid:
                    fid.write(b"payload%04d" % i)
                link = os.path.join(self.root, "annexlike", "f%d.bin" % i)
                if not os.path.lexists(link):
                    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
                files.append(_walk_one(link, "f%d.bin" % i))
            self.assertEqual(_prehash(files, cas, 4), 6)
            self.assertEqual(cas.memo_count(), 6)
        finally:
            cas.close()

    def test_dangling_links_are_skipped_by_prehash(self):
        from jdata.njbids import _prehash

        cas = CAS(os.path.join(self.root, "cas5"), commit_every=1)
        try:
            link = os.path.join(self.root, "gone.bin")
            os.symlink(
                "../.git/annex/objects/aa/bb/MD5E-s9--%032d.bin/MD5E-s9--%032d.bin" % (0, 0),
                link,
            )
            self.assertEqual(_prehash([_walk_one(link, "gone.bin")], cas, 2), 0)
        finally:
            cas.close()


class TestTrailingCommaRepair(unittest.TestCase):
    """The repair must be string-aware, or it silently corrupts data."""

    def test_removes_trailing_comma_before_bracket_and_brace(self):
        self.assertEqual(strip_trailing_commas("[1,2,]"), "[1,2]")
        self.assertEqual(strip_trailing_commas('{"a":1,}'), '{"a":1}')

    def test_handles_whitespace_and_newlines_before_the_closer(self):
        self.assertEqual(strip_trailing_commas("[1,\n  2,\n]"), "[1,\n  2\n]")

    def test_nested_containers(self):
        text = '{"a": [1, 2,], "b": {"c": 3,},}'
        self.assertEqual(json.loads(strip_trailing_commas(text)), {"a": [1, 2], "b": {"c": 3}})

    def test_comma_inside_a_string_is_preserved(self):
        for text in ('{"a": "x,]"}', '{"a": "y,}"}', '["p,]","q,}"]'):
            self.assertEqual(json.loads(strip_trailing_commas(text)), json.loads(text))

    def test_escaped_quote_does_not_break_string_tracking(self):
        text = '{"a": "he said \\"hi,]\\"", "b": [1,]}'
        self.assertEqual(
            json.loads(strip_trailing_commas(text)),
            json.loads('{"a": "he said \\"hi,]\\"", "b": [1]}'),
        )

    def test_valid_json_is_returned_unchanged(self):
        for text in ('{"a":[1,2]}', "[]", "{}", '{"a": {"b": [1, 2, 3]}}'):
            self.assertEqual(strip_trailing_commas(text), text)

    def test_legitimate_commas_between_items_are_kept(self):
        self.assertEqual(strip_trailing_commas("[1, 2, 3]"), "[1, 2, 3]")


class TestManifestIdempotency(unittest.TestCase):
    """A file must appear in the manifest exactly once, however it is handled.

    Regression test: a format handler registers the file before parsing it, and
    a parse failure falls through to the generic link branch, which registered
    it a second time.  ds006391 ended up with 98 duplicated manifest paths --
    inflating the reported file count and corrupting the fingerprint, which is
    computed over the manifest.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = make_bids(os.path.join(self.root, "dsMI"), git=True, derivatives=False)
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _convert(self):
        return bids2json(self.ds, dbname="db", dsname="dsMI", cas=self.cas)

    def test_no_duplicate_paths_when_a_handler_fails(self):
        # a .mat that is neither MATLAB nor VEST: registered, then falls through
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_model.mat")
        with open(target, "wb") as fid:
            fid.write(b"\x00\x01\x02 not a mat file at all" * 10)
        result = self._convert()
        paths = [e["path"] for e in result["manifest"]]
        self.assertEqual(len(paths), len(set(paths)))

    def test_reported_file_count_matches_unique_paths(self):
        result = self._convert()
        paths = {e["path"] for e in result["manifest"]}
        self.assertEqual(result["doc"][".neurojson"]["Files"], len(paths))

    def test_fsl_vest_matrix_yields_searchable_metadata(self):
        target = os.path.join(self.ds, "sub-01", "anat", "sub-01_design.mat")
        with open(target, "w") as fid:
            fid.write("/NumWaves 4\n/NumPoints 3\n/Matrix\n1 2 3 4\n5 6 7 8\n9 10 11 12\n")
        result = self._convert()
        node = result["doc"]["sub-01"]["anat"]["sub-01_design.mat"]
        self.assertEqual(node["VESTHeader"]["NumWaves"], 4)
        self.assertIn("_DataLink_", node["MATObject"])
        self.assertEqual(result["errors"], [])

    def test_offloaded_file_is_not_manifested_twice(self):
        rows = ["onset\tduration\ttrial_type"]
        for i in range(8000):
            rows.append("%f\t1.0\tcondition_%d" % (i * 0.5, i % 5))
        with open(
            os.path.join(self.ds, "sub-01", "func", "sub-01_task-rest_events.tsv"), "w"
        ) as fid:
            fid.write("\n".join(rows) + "\n")
        result = bids2json(
            self.ds,
            dbname="db",
            dsname="dsMI",
            cas=self.cas,
            max_doc=40000,
            max_tsv=1 << 30,
        )
        self.assertTrue(result["stats"]["offloaded"])
        paths = [e["path"] for e in result["manifest"]]
        self.assertEqual(len(paths), len(set(paths)))


class TestBudgetIsStrict(unittest.TestCase):
    """The budget must be an actual ceiling, not an estimate.

    Regression test: the offload loop tracked size by decrementing a running
    total, which drifts from the true serialised length because removing a node
    also shifts separators and key ordering.  ds003097 came out 211 bytes over
    its 7,500,000 budget.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ds = os.path.join(self.root, "dsSB2")
        make_bids(self.ds, subjects=("01",), git=True, derivatives=False)
        for sub in ("02", "03", "04", "05"):
            for i in range(40):
                make_nifti(
                    os.path.join(
                        self.ds,
                        "sub-%s" % sub,
                        "func",
                        "sub-%s_task-rest_run-%03d_bold.nii.gz" % (sub, i),
                    ),
                    dims=(2, 2, 2),
                )
                with open(
                    os.path.join(
                        self.ds,
                        "sub-%s" % sub,
                        "func",
                        "sub-%s_task-rest_run-%03d_events.tsv" % (sub, i),
                    ),
                    "w",
                ) as fid:
                    fid.write("onset\tduration\n" + "".join("%d\t1\n" % j for j in range(40)))
        self.cas = CAS(os.path.join(self.root, "cas"), commit_every=1)

    def tearDown(self):
        self.cas.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_document_never_exceeds_the_budget(self):
        for budget in (3000, 8000, 20000, 60000):
            result = bids2json(self.ds, dbname="db", dsname="dsSB2", cas=self.cas, max_doc=budget)
            actual = len(canonical_json(result["doc"]))
            self.assertLessEqual(
                actual, budget, "budget %d exceeded by %d bytes" % (budget, actual - budget)
            )

    def test_tier_one_is_capped_so_wide_datasets_shed_subtrees(self):
        """Leaf-by-leaf offloading degenerates on a wide dataset."""
        result = bids2json(
            self.ds,
            dbname="db",
            dsname="dsSB2",
            cas=self.cas,
            max_doc=4000,
            max_tsv=1 << 30,
            max_leaf_offloads=4,
        )
        how = collections.Counter(item["how"] for item in result["stats"]["offloaded"])
        self.assertLessEqual(how["leaf"], 4)
        self.assertGreater(how["subtree"], 0)
        self.assertLessEqual(len(canonical_json(result["doc"])), 4000)

    def test_capped_run_is_still_deterministic(self):
        args = dict(dbname="db", dsname="dsSB2", cas=self.cas, max_doc=4000, max_leaf_offloads=4)
        one = bids2json(self.ds, **args)
        two = bids2json(self.ds, **args)
        self.assertEqual(one["fingerprint"], two["fingerprint"])
        self.assertEqual(canonical_json(one["doc"]), canonical_json(two["doc"]))
