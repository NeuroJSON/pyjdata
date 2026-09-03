"""Tests for the CouchDB design document in neurojson/design/qq.

A design document is JavaScript run by the database server, so a mistake in a
map function normally surfaces only after documents are published and the index
is rebuilt.  These tests evaluate the real view sources against a real converted
document via neurojson/simulate.js, which needs node but no CouchDB.

Two compatibility contracts are asserted explicitly, because breaking either one
silently degrades the search layer rather than raising an error:

  * ``subjects`` must emit the 7-element key the Postgres sync slices, with age
    and sex extracted from ``participants.tsv``
  * ``links`` must emit ``[doc._id, ext, size]`` and an ``ext`` that survives the
    sync's ``isValidFileType()`` filter
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)  # so the shared BIDS fixture builder is importable

from jdata.njcas import CAS
from jdata.njbids import bids2json, canonical_json
from jdata.njcouch import design_from_dir

from testnjbids import make_bids  # noqa: E402  (shared fixture builder)

DESIGN = os.path.join(ROOT, "neurojson", "design", "qq")
SIMULATE = os.path.join(ROOT, "neurojson", "simulate.js")


def have_node():
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def run_view(view, docpath):
    out = subprocess.run(
        ["node", SIMULATE, "view", DESIGN, view, docpath],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AssertionError("simulate.js failed: %s" % out.stderr[:2000])
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def run_update(handler, docpath):
    out = subprocess.run(
        ["node", SIMULATE, "update", DESIGN, handler, docpath],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AssertionError("simulate.js failed: %s" % out.stderr[:2000])
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


class TestDesignAssembly(unittest.TestCase):
    def test_design_dir_assembles(self):
        ddoc = design_from_dir(DESIGN)
        self.assertEqual(ddoc["language"], "javascript")
        for view in ("dbinfo", "subjects", "links", "participantsfields", "updatetime", "versions"):
            self.assertIn(view, ddoc["views"])
            self.assertIn("map", ddoc["views"][view])
        self.assertIn("timestamp", ddoc["updates"])
        self.assertIn("validate_doc_update", ddoc)

    def test_metadata_key_is_neurojson_everywhere(self):
        """The legacy '.datainfo' key must not survive anywhere in the new design."""
        for name in os.listdir(DESIGN):
            with open(os.path.join(DESIGN, name), encoding="utf-8") as fid:
                self.assertNotIn(".datainfo", fid.read(), "%s still uses .datainfo" % name)


@unittest.skipUnless(have_node(), "node is required to evaluate view JavaScript")
class TestViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        ds = make_bids(os.path.join(cls.root, "dsV"), subjects=("01", "02", "03"), git=True)
        cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        result = bids2json(ds, dbname="testdb", dsname="dsV", cas=cas)
        cas.close()
        cls.result = result
        cls.docdir = os.path.join(cls.root, "out", "dsV", "1.0.0")
        os.makedirs(cls.docdir)
        cls.docpath = os.path.join(cls.docdir, "doc.json")
        with open(cls.docpath, "w", encoding="utf-8") as fid:
            fid.write(canonical_json(result["doc"]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_all_views_execute_without_error(self):
        for view in ("dbinfo", "subjects", "links", "participantsfields", "updatetime", "versions"):
            run_view(view, self.docpath)  # raises on failure

    def test_dbinfo_emits_one_row_with_version_metadata(self):
        rows = run_view("dbinfo", self.docpath)
        self.assertEqual(len(rows), 1)
        value = rows[0]["value"]
        self.assertEqual(rows[0]["id"], "dsV")
        self.assertEqual(value["name"], "Synthetic Test Dataset")
        self.assertEqual(sorted(value["subj"]), ["sub-01", "sub-02", "sub-03"])
        self.assertEqual(sorted(value["modality"]), ["anat", "dwi", "func"])
        # Version is empty because this fixture is not on a release tag
        self.assertEqual(value["version"], "")
        self.assertTrue(value["label"].startswith("2.3.1+g"))
        self.assertTrue(value["commit"])
        self.assertTrue(value["commit"])
        self.assertGreater(value["files"], 0)

    def test_subjects_key_shape_matches_postgres_sync(self):
        rows = run_view("subjects", self.docpath)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row["key"]), 7)
            age, sex = row["key"][0], row["key"][1]
            self.assertEqual(len(age), 5)
            self.assertTrue(age.isdigit(), age)
            self.assertEqual(len(sex), 4)
        ages = sorted(row["key"][0] for row in rows)
        # ages 20, 21, 22 are stored as age*100 zero-padded to 5 digits
        self.assertEqual(ages, ["02000", "02100", "02200"])
        sexes = sorted(row["key"][1] for row in rows)
        self.assertEqual(sexes, ["000F", "000F", "000M"])

    def test_subjects_value_lists_modalities_and_tasks(self):
        rows = run_view("subjects", self.docpath)
        value = rows[0]["value"]
        self.assertEqual(sorted(value["modalities"]), ["anat", "dwi", "func"])
        self.assertEqual(value["tasks"], ["rest"])
        self.assertIn("T1w", value["types"])

    def test_links_key_is_id_first_and_carries_the_hash(self):
        rows = run_view("links", self.docpath)
        self.assertTrue(rows)
        for row in rows:
            docid, ext, size = row["key"]
            self.assertEqual(docid, "dsV")
            self.assertTrue(ext.startswith("."), ext)
            self.assertIsInstance(size, int)
            self.assertEqual(row["value"]["algo"], "sha256")
            self.assertEqual(len(row["value"]["hash"]), 64)

    def test_links_extensions_pass_the_sync_validity_filter(self):
        """Mirrors isValidFileType() in backend/sync/incrementalSync.js."""
        for row in run_view("links", self.docpath):
            ext = row["key"][1]
            self.assertTrue(ext.startswith("."))
            self.assertNotIn("/", ext)
            self.assertLessEqual(len(ext), 20)

    def test_links_are_deduplicated_by_url(self):
        rows = run_view("links", self.docpath)
        urls = [row["value"]["url"] for row in rows]
        self.assertEqual(len(urls), len(set(urls)))

    def test_links_path_is_a_usable_jsonpath(self):
        rows = run_view("links", self.docpath)
        for row in rows:
            self.assertTrue(row["value"]["path"].startswith("$."))

    def test_versions_view_reports_the_upstream_identifiers(self):
        rows = run_view("versions", self.docpath)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["key"][0], "dsV")
        self.assertTrue(rows[0]["key"][1].startswith("2.3.1+g"))
        self.assertEqual(rows[0]["value"]["commit"], self.result["version"]["SourceCommit"])

    def test_updatetime_is_empty_before_publication(self):
        """UpdateTime is stamped by the server, so it is absent on disk."""
        self.assertEqual(run_view("updatetime", self.docpath), [])

    def test_participantsfields_lists_columns(self):
        rows = run_view("participantsfields", self.docpath)
        self.assertEqual(len(rows), 1)
        self.assertEqual(sorted(rows[0]["key"]), ["age", "handedness", "participant_id", "sex"])


@unittest.skipUnless(have_node(), "node is required to evaluate handler JavaScript")
class TestUpdateHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        ds = make_bids(os.path.join(cls.root, "dsU"), git=True, derivatives=False)
        cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        result = bids2json(ds, dbname="testdb", dsname="dsU", cas=cas)
        cas.close()
        cls.docdir = os.path.join(cls.root, "out", "dsU", "2.3.1")
        os.makedirs(cls.docdir)
        cls.docpath = os.path.join(cls.docdir, "doc.json")
        with open(cls.docpath, "w", encoding="utf-8") as fid:
            fid.write(canonical_json(result["doc"]))
        cls.version = result["version"]["Version"]
        cls.pushes = run_update("timestamp", cls.docpath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_three_pushes_all_succeed(self):
        self.assertEqual([p["push"] for p in self.pushes], [1, 2, 3])
        for push in self.pushes:
            self.assertEqual(push["id"], "dsU")

    def test_createtime_is_set_once_and_never_moves(self):
        created = [p["meta"]["CreateTime"] for p in self.pushes]
        self.assertEqual(len(set(created)), 1)
        self.assertGreater(created[0], 0)

    def test_updatetime_advances_on_every_push(self):
        stamps = [p["meta"]["UpdateTime"] for p in self.pushes]
        self.assertEqual(stamps, sorted(stamps))
        self.assertGreater(stamps[-1], stamps[0])

    def test_converter_metadata_is_preserved_through_the_merge(self):
        meta = self.pushes[-1]["meta"]
        self.assertTrue(meta["VersionLabel"].startswith("2.3.1+g"))
        self.assertTrue(meta["SourceCommit"])
        self.assertGreater(meta["Files"], 0)

    def test_metadata_block_sorts_first(self):
        """Keeps '.neurojson' visible at the top of the document in a browser."""
        for push in self.pushes:
            self.assertEqual(push["topkeys"][0], "_id")
            self.assertEqual(push["topkeys"][1], ".neurojson")

    def test_client_cannot_forge_the_timestamps(self):
        """A pushed body carrying CreateTime/UpdateTime must be ignored."""
        forged = os.path.join(self.docdir, "forged.json")
        with open(self.docpath, encoding="utf-8") as fid:
            doc = json.load(fid)
        doc[".neurojson"]["CreateTime"] = 1.0
        doc[".neurojson"]["UpdateTime"] = 2.0
        doc["_id"] = "dsU"
        with open(forged, "w", encoding="utf-8") as fid:
            json.dump(doc, fid)
        pushes = run_update("timestamp", forged)
        for push in pushes:
            self.assertNotEqual(push["meta"]["CreateTime"], 1.0)
            self.assertNotEqual(push["meta"]["UpdateTime"], 2.0)


if __name__ == "__main__":
    unittest.main()


PGSYNC_ENV = "NEUROJSON_PGSYNC"  # path to backend/sync/incrementalSync.js
PGCOMPAT = os.path.join(ROOT, "neurojson", "pgcompat.js")


def _pgsync_path():
    """Locate the Postgres sync source, if the backend checkout is available."""
    explicit = os.environ.get(PGSYNC_ENV)
    if explicit and os.path.isfile(explicit):
        return explicit
    return None


def run_pg(transform, docpath, syncfile):
    out = subprocess.run(
        ["node", PGCOMPAT, syncfile, transform, docpath],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AssertionError("pgcompat.js failed: %s" % out.stderr[:2000])
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


@unittest.skipUnless(have_node(), "node is required")
@unittest.skipUnless(_pgsync_path(), "set %s to backend/sync/incrementalSync.js" % PGSYNC_ENV)
class TestPostgresSyncCompatibility(unittest.TestCase):
    """The Postgres search layer keeps its own ports of the CouchDB views.

    ``backend/sync/incrementalSync.js`` reimplements the dbinfo and subjects map
    functions in Node so that an incremental sync needs two HTTP requests
    instead of three, and its own comment notes that those copies "drift
    silently" from the originals.  A document schema change can therefore keep
    CouchDB perfectly happy while quietly emptying out Postgres search results,
    so the two implementations are compared directly here.
    """

    @classmethod
    def setUpClass(cls):
        cls.sync = _pgsync_path()
        cls.root = tempfile.mkdtemp()
        ds = make_bids(os.path.join(cls.root, "dsP"), subjects=("01", "02", "03", "04"), git=True)
        cas = CAS(os.path.join(cls.root, "cas"), commit_every=1)
        cls.result = bids2json(ds, dbname="testdb", dsname="dsP", cas=cas)
        cas.close()
        cls.docdir = os.path.join(cls.root, "out", "dsP", "2.3.1")
        os.makedirs(cls.docdir)
        cls.docpath = os.path.join(cls.docdir, "doc.json")
        with open(cls.docpath, "w", encoding="utf-8") as fid:
            fid.write(canonical_json(cls.result["doc"]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_subjects_rows_match_the_couchdb_view_exactly(self):
        pg = run_pg("subjects", self.docpath, self.sync)
        view = run_view("subjects", self.docpath)
        self.assertEqual(len(pg), len(view))
        self.assertEqual(
            {tuple(r["key"]): r["value"] for r in pg},
            {tuple(r["key"]): r["value"] for r in view},
        )

    def test_subjects_key_is_sliceable_the_way_the_sync_slices_it(self):
        # firstSync() reads row.key[6] as the ioviews.subj column
        for row in run_pg("subjects", self.docpath, self.sync):
            self.assertTrue(row["subj"])
            self.assertEqual(row["view"], "subjects")

    def test_dbinfo_is_produced_and_names_the_dataset(self):
        rows = run_pg("dbinfo", self.docpath, self.sync)
        self.assertEqual(len(rows), 1)
        value = rows[0]["value"]
        self.assertEqual(value["name"], "Synthetic Test Dataset")
        self.assertEqual(rows[0]["subj"], "4")
        self.assertGreater(value["length"], 0)

    def test_document_is_valid_postgres_jsonb(self):
        """jsonb rejects \\u0000, and the sync only strips it from its own output."""
        with open(self.docpath, encoding="utf-8") as fid:
            text = fid.read()
        self.assertNotIn("\\u0000", text)
        self.assertNotIn("\x00", text)

    def test_ioviews_key_columns_are_present_for_every_row(self):
        """ioviews has UNIQUE(dbname, dsname, subj, view); none may be null."""
        rows = run_pg("subjects", self.docpath, self.sync) + run_pg(
            "dbinfo", self.docpath, self.sync
        )
        for row in rows:
            self.assertTrue(row["id"])
            self.assertTrue(row["view"])
            self.assertIsNotNone(row["subj"])
