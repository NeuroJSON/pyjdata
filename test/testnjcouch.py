"""Tests for jdata.njcouch -- the CouchDB publication client.

Network-free by default: a stub HTTP handler records what the client actually
sends.  The contract being pinned down is that *data* documents go out as a POST
to the update handler and never as a PUT, because the handler is what resolves
``_rev`` and owns the timestamp metadata.

Set ``NEUROJSON_TEST_SERVER`` to also run the read-only live checks.
"""

import os
import io
import sys
import json
import shutil
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from jdata.njcouch import CouchDB, CouchError, design_from_dir


class _Recorder:
    """Replaces urlopen and records (method, url, headers, body)."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, req, timeout=None, context=None):
        body = req.data.decode("utf-8") if req.data else None
        self.calls.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "headers": dict(req.header_items()),
                "body": body,
            }
        )
        status, payload = self.responses.get((req.get_method(), req.full_url), (200, {"ok": True}))
        if status >= 400:
            raise urllib.error.HTTPError(
                req.full_url, status, "err", {}, io.BytesIO(json.dumps(payload).encode())
            )
        return _Response(status, payload)


class _Response:
    def __init__(self, status, payload):
        self.status = status
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CouchTestCase(unittest.TestCase):
    def setUp(self):
        self.rec = _Recorder()
        self._orig = urllib.request.urlopen
        urllib.request.urlopen = self.rec
        self.couch = CouchDB("http://example.invalid:5984", user="u", password="p", retries=0)

    def tearDown(self):
        urllib.request.urlopen = self._orig


class TestCredentialHandling(unittest.TestCase):
    def test_credentials_are_stripped_from_the_stored_url(self):
        couch = CouchDB("https://admin:secret@host.example:7777")
        self.assertEqual(couch.url, "https://host.example:7777")
        self.assertEqual(couch.user, "admin")
        self.assertEqual(couch.password, "secret")

    def test_url_without_credentials_is_untouched(self):
        couch = CouchDB("http://host.example:5984")
        self.assertEqual(couch.url, "http://host.example:5984")
        self.assertIsNone(couch.user)

    def test_trailing_slash_is_normalised(self):
        self.assertEqual(CouchDB("http://h:5984/").url, "http://h:5984")

    def test_error_message_does_not_leak_the_password(self):
        couch = CouchDB("https://admin:supersecret@host.example:7777")
        err = CouchError(500, {"error": "boom"}, couch.url + "/db")
        self.assertNotIn("supersecret", str(err))


class TestPushSemantics(CouchTestCase):
    def test_push_uses_post_to_the_update_handler(self):
        self.couch.push("mydb", "ds000001", {"a": 1})
        call = self.rec.calls[-1]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["url"],
            "http://example.invalid:5984/mydb/_design/qq/_update/timestamp/ds000001",
        )
        self.assertEqual(json.loads(call["body"]), {"a": 1})

    def test_push_never_issues_a_put(self):
        self.couch.push("mydb", "ds1", {"a": 1})
        self.assertNotIn("PUT", [c["method"] for c in self.rec.calls])

    def test_push_sends_basic_auth(self):
        self.couch.push("mydb", "ds1", {"a": 1})
        headers = {k.lower(): v for k, v in self.rec.calls[-1]["headers"].items()}
        self.assertTrue(headers["authorization"].startswith("Basic "))

    def test_docid_is_url_escaped(self):
        self.couch.push("my db", "ds/1", {})
        self.assertIn("/my%20db/", self.rec.calls[-1]["url"])
        self.assertTrue(self.rec.calls[-1]["url"].endswith("/ds%2F1"))

    def test_push_file_sends_the_bytes_verbatim(self):
        """The published bytes must be exactly the canonical, fingerprinted bytes."""
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "doc.json")
            payload = '{"b":1,"a":{"c":3}}'
            with open(path, "w") as fid:
                fid.write(payload)
            self.couch.push_file("mydb", "ds1", path)
            self.assertEqual(self.rec.calls[-1]["body"], payload)
            self.assertEqual(self.rec.calls[-1]["method"], "POST")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_custom_design_and_handler_names(self):
        self.couch.push("db", "d", {}, design="zz", handler="stamp")
        self.assertIn("/_design/zz/_update/stamp/d", self.rec.calls[-1]["url"])

    def test_failed_push_raises_with_status(self):
        url = "http://example.invalid:5984/db/_design/qq/_update/timestamp/d"
        self.rec.responses[("POST", url)] = (403, {"error": "forbidden"})
        with self.assertRaises(CouchError) as ctx:
            self.couch.push("db", "d", {})
        self.assertEqual(ctx.exception.status, 403)


class TestDesignDocuments(CouchTestCase):
    def test_put_design_fetches_current_rev_first(self):
        base = "http://example.invalid:5984/db/_design/qq"
        self.rec.responses[("GET", base)] = (200, {"_id": "_design/qq", "_rev": "7-abc"})
        self.couch.put_design("db", {"views": {}}, name="qq")
        methods = [c["method"] for c in self.rec.calls]
        self.assertEqual(methods, ["GET", "PUT"])
        self.assertEqual(json.loads(self.rec.calls[-1]["body"])["_rev"], "7-abc")

    def test_put_design_omits_rev_when_absent(self):
        base = "http://example.invalid:5984/db/_design/qq"
        self.rec.responses[("GET", base)] = (404, {"error": "not_found"})
        self.couch.put_design("db", {"views": {}})
        self.assertNotIn("_rev", json.loads(self.rec.calls[-1]["body"]))

    def test_view_query_json_encodes_non_string_params(self):
        self.couch.view("db", "links", startkey=["ds1"], endkey=["ds1", {}], limit="5")
        url = self.rec.calls[-1]["url"]
        self.assertIn("startkey=%5B%22ds1%22%5D", url)
        self.assertIn("limit=5", url)


class TestDesignFromDir(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, text="function(doc){}"):
        with open(os.path.join(self.root, name), "w") as fid:
            fid.write(text)

    def test_file_naming_maps_to_design_structure(self):
        self._write("view_a.js")
        self._write("reduce_a.js", "_count")
        self._write("view_b.js")
        self._write("update_stamp.js")
        self._write("filter_only.js")
        self._write("validate_doc_update.js")
        self._write("notes.txt", "ignored")
        ddoc = design_from_dir(self.root)
        self.assertEqual(sorted(ddoc["views"]), ["a", "b"])
        self.assertEqual(ddoc["views"]["a"]["reduce"], "_count")
        self.assertNotIn("reduce", ddoc["views"]["b"])
        self.assertIn("stamp", ddoc["updates"])
        self.assertIn("only", ddoc["filters"])
        self.assertIn("validate_doc_update", ddoc)

    def test_empty_sections_are_omitted(self):
        self._write("view_a.js")
        ddoc = design_from_dir(self.root)
        self.assertNotIn("updates", ddoc)
        self.assertNotIn("filters", ddoc)

    def test_real_design_dir_round_trips(self):
        ddoc = design_from_dir(os.path.join(ROOT, "neurojson", "design", "qq"))
        self.assertIn("dbinfo", ddoc["views"])
        self.assertIn("timestamp", ddoc["updates"])


class TestRetry(unittest.TestCase):
    def setUp(self):
        self._orig = urllib.request.urlopen

    def tearDown(self):
        urllib.request.urlopen = self._orig

    def test_server_error_is_retried_then_returned(self):
        attempts = []

        def flaky(req, timeout=None, context=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                req.full_url, 503, "busy", {}, io.BytesIO(b'{"error":"busy"}')
            )

        urllib.request.urlopen = flaky
        couch = CouchDB("http://h:5984", retries=2)
        couch.request  # noqa: B018
        import jdata.njcouch as mod

        orig_sleep = mod.time.sleep
        mod.time.sleep = lambda _s: None
        try:
            status, _ = couch.request("GET", "/db")
        finally:
            mod.time.sleep = orig_sleep
        self.assertEqual(status, 503)
        self.assertEqual(len(attempts), 3)  # initial try plus two retries

    def test_client_error_is_not_retried(self):
        attempts = []

        def denied(req, timeout=None, context=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                req.full_url, 401, "no", {}, io.BytesIO(b'{"error":"unauthorized"}')
            )

        urllib.request.urlopen = denied
        status, _ = CouchDB("http://h:5984", retries=3).request("GET", "/db")
        self.assertEqual(status, 401)
        self.assertEqual(len(attempts), 1)


@unittest.skipUnless(os.environ.get("NEUROJSON_TEST_SERVER"), "no live server configured")
class TestLiveServerReadOnly(unittest.TestCase):
    """Read-only checks against a real CouchDB (set NEUROJSON_TEST_SERVER)."""

    def setUp(self):
        self.couch = CouchDB(
            os.environ["NEUROJSON_TEST_SERVER"],
            netrc_machine=os.environ.get("NEUROJSON_TEST_NETRC", "neurojson.io"),
        )

    def test_welcome(self):
        self.assertIn("couchdb", self.couch.welcome())

    def test_session_reports_a_user(self):
        self.assertIn("userCtx", self.couch.session())

    def test_all_dbs_is_a_list(self):
        self.assertIsInstance(self.couch.all_dbs(), list)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    os.environ.get("NEUROJSON_TEST_SERVER") and os.environ.get("NEUROJSON_TEST_DB"),
    "set NEUROJSON_TEST_SERVER and NEUROJSON_TEST_DB (a scratch database) to run",
)
class TestLivePushSemantics(unittest.TestCase):
    """Verify the publish contract against a real CouchDB.

    The node simulator in test/testnjviews.py checks the handler *logic*; this
    checks that CouchDB itself compiles the design document, that a POST to the
    update handler resolves ``_rev`` server-side, and that the timestamp
    metadata behaves as designed.  Those are the parts that would only fail at
    deploy time.

    Creates a temporary design document and one document, both prefixed
    ``njtest``, and removes them again in tearDownClass.
    """

    DDOC = "njtest"
    DOCID = "njtest-push-probe"

    @classmethod
    def setUpClass(cls):
        from jdata.njcouch import design_from_dir

        cls.db = os.environ["NEUROJSON_TEST_DB"]
        cls.couch = CouchDB(
            os.environ["NEUROJSON_TEST_SERVER"],
            netrc_machine=os.environ.get("NEUROJSON_TEST_NETRC", "neurojson.io"),
        )
        cls.couch.put_design(
            cls.db, design_from_dir(os.path.join(ROOT, "neurojson", "design", "qq")), name=cls.DDOC
        )
        cls.doc = {
            ".neurojson": {
                "Version": "1.2.3",
                "VersionSource": "git-tag",
                "SourceCommit": "0" * 40,
                "Fingerprint": "a" * 64,
                "Tags": ["1.2.3"],
                "Files": 2,
                "Bytes": 1234,
            },
            "dataset_description.json": {"Name": "njtest probe", "BIDSVersion": "1.8.0"},
            "README": "temporary test document",
            "participants.tsv": {"participant_id": ["sub-01"], "age": [33], "sex": ["F"]},
            "sub-01": {
                "anat": {
                    "sub-01_T1w.nii.gz": {
                        "_DataLink_": (
                            "https://neurojson.io/io/cas.cgi?action=get&db=%s&doc=%s"
                            "&hash=sha256:%s&size=99&file=sub-01/anat/sub-01_T1w.nii.gz"
                        )
                        % (cls.db, cls.DOCID, "b" * 64)
                    }
                }
            },
        }

    @classmethod
    def tearDownClass(cls):
        for remove in (
            lambda: cls.couch.delete_doc(cls.db, cls.DOCID),
            lambda: cls.couch.request(
                "DELETE",
                "/%s/_design/%s?rev=%s"
                % (cls.db, cls.DDOC, cls.couch.get_design(cls.db, cls.DDOC)["_rev"]),
            ),
        ):
            try:
                remove()
            except Exception:
                pass

    def test_push_creates_then_updates_with_stable_createtime(self):
        self.couch.push(self.db, self.DOCID, self.doc, design=self.DDOC)
        first = self.couch.get_doc(self.db, self.DOCID)
        time.sleep(1.1)
        self.couch.push(self.db, self.DOCID, self.doc, design=self.DDOC)
        second = self.couch.get_doc(self.db, self.DOCID)

        self.assertNotEqual(first["_rev"], second["_rev"])
        self.assertEqual(first[".neurojson"]["CreateTime"], second[".neurojson"]["CreateTime"])
        self.assertGreater(second[".neurojson"]["UpdateTime"], first[".neurojson"]["UpdateTime"])
        self.assertEqual(second[".neurojson"]["Version"], "1.2.3")
        self.assertEqual(second[".neurojson"]["Fingerprint"], "a" * 64)

    def test_server_ignores_client_supplied_timestamps(self):
        self.couch.push(self.db, self.DOCID, self.doc, design=self.DDOC)
        genuine = self.couch.get_doc(self.db, self.DOCID)[".neurojson"]["CreateTime"]
        forged = dict(self.doc)
        forged[".neurojson"] = dict(self.doc[".neurojson"], CreateTime=1.0, UpdateTime=2.0)
        self.couch.push(self.db, self.DOCID, forged, design=self.DDOC)
        after = self.couch.get_doc(self.db, self.DOCID)[".neurojson"]
        self.assertEqual(after["CreateTime"], genuine)
        self.assertGreater(after["UpdateTime"], 2.0)

    def test_views_execute_inside_couchdb(self):
        self.couch.push(self.db, self.DOCID, self.doc, design=self.DDOC)
        for view in ("dbinfo", "subjects", "links", "versions", "updatetime"):
            res = self.couch.view(self.db, view, design=self.DDOC, limit=500)
            rows = [r for r in res["rows"] if r["id"] == self.DOCID]
            self.assertEqual(len(rows), 1, "view %s produced %d rows" % (view, len(rows)))
        links = [
            r
            for r in self.couch.view(self.db, "links", design=self.DDOC, limit=500)["rows"]
            if r["id"] == self.DOCID
        ]
        self.assertEqual(links[0]["key"][1], ".nii.gz")
        self.assertEqual(links[0]["value"]["algo"], "sha256")
        self.assertEqual(links[0]["value"]["hash"], "b" * 64)


class TestPutDocIsSeparateFromPush(unittest.TestCase):
    """Configuration documents take a different path from dataset digests."""

    def setUp(self):
        self.rec = _Recorder()
        self._orig = urllib.request.urlopen
        urllib.request.urlopen = self.rec
        self.couch = CouchDB("http://example.invalid:5984", user="u", password="p", retries=0)

    def tearDown(self):
        urllib.request.urlopen = self._orig

    def test_put_doc_preserves_a_supplied_rev(self):
        self.couch.put_doc("sys", "registry", {"_rev": "58-abc", "database": []})
        call = self.rec.calls[-1]
        self.assertEqual(call["method"], "PUT")
        self.assertEqual(json.loads(call["body"])["_rev"], "58-abc")

    def test_put_doc_fetches_rev_when_not_supplied(self):
        url = "http://example.invalid:5984/sys/registry"
        self.rec.responses[("GET", url)] = (200, {"_id": "registry", "_rev": "9-zzz"})
        self.couch.put_doc("sys", "registry", {"database": []})
        self.assertEqual([c["method"] for c in self.rec.calls], ["GET", "PUT"])
        self.assertEqual(json.loads(self.rec.calls[-1]["body"])["_rev"], "9-zzz")

    def test_put_doc_omits_rev_for_a_new_document(self):
        url = "http://example.invalid:5984/sys/newdoc"
        self.rec.responses[("GET", url)] = (404, {"error": "not_found"})
        self.couch.put_doc("sys", "newdoc", {"a": 1})
        self.assertNotIn("_rev", json.loads(self.rec.calls[-1]["body"]))

    def test_dataset_push_still_never_uses_put(self):
        self.couch.push("openneuro_full", "ds000001", {"a": 1})
        self.assertEqual([c["method"] for c in self.rec.calls], ["POST"])


class TestOversizedBodyHandling(unittest.TestCase):
    """An oversized document is reported two different ways by CouchDB.

    A PUT gets a clean ``413 document_too_large``.  A POST to an update handler
    gets its connection closed instead, which on its own is indistinguishable
    from the server having gone away -- so the publish path checks whether the
    server is still answering before deciding.  Re-uploading many megabytes
    several times to rediscover a rejection is also pure waste, so transport
    failures are only retried for small requests.
    """

    def setUp(self):
        self._orig = urllib.request.urlopen

    def tearDown(self):
        urllib.request.urlopen = self._orig

    def test_large_body_transport_failure_is_not_retried(self):
        attempts = []

        def reset(req, timeout=None, context=None):
            attempts.append(len(req.data or b""))
            raise ConnectionResetError(104, "Connection reset by peer")

        urllib.request.urlopen = reset
        couch = CouchDB("http://h:5984", retries=4, retry_body_limit=1 << 20)
        with self.assertRaises(CouchError):
            couch.request("POST", "/db/_design/qq/_update/timestamp/d", raw=b"x" * (4 << 20))
        self.assertEqual(len(attempts), 1)

    def test_small_body_transport_failure_is_retried(self):
        attempts = []

        def reset(req, timeout=None, context=None):
            attempts.append(1)
            raise ConnectionResetError(104, "Connection reset by peer")

        urllib.request.urlopen = reset
        import jdata.njcouch as mod

        orig_sleep = mod.time.sleep
        mod.time.sleep = lambda _s: None
        try:
            couch = CouchDB("http://h:5984", retries=2, retry_body_limit=1 << 20)
            with self.assertRaises(CouchError):
                couch.request("POST", "/db/d", raw=b"tiny")
        finally:
            mod.time.sleep = orig_sleep
        self.assertEqual(len(attempts), 3)

    def test_alive_reports_true_when_the_server_answers(self):
        rec = _Recorder({("GET", "http://h:5984/_up"): (200, {"status": "ok"})})
        urllib.request.urlopen = rec
        self.assertTrue(CouchDB("http://h:5984", retries=0).alive())

    def test_alive_reports_false_when_it_does_not(self):
        def dead(req, timeout=None, context=None):
            raise ConnectionResetError(104, "gone")

        urllib.request.urlopen = dead
        self.assertFalse(CouchDB("http://h:5984", retries=0).alive())
