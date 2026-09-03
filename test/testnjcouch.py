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
