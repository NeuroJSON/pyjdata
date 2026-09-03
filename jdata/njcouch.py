"""
CouchDB client for publishing NeuroJSON dataset digests.

Documents are never written with a bare ``PUT``.  They are POSTed to a server
side *update handler* (``_design/<design>/_update/<handler>/<docid>``), which

* resolves ``_rev`` on the server, so a client never has to fetch-then-write and
  cannot lose a concurrent update to a stale revision, and
* stamps the metadata block (``.neurojson.CreateTime`` on first insert,
  ``.neurojson.UpdateTime`` on every write) with the *server's* clock.

The second point is what keeps the digest version-invariant: if the converter
wrote its own timestamps, the document bytes would change on every run and the
content fingerprint would be worthless.  Keeping them server-side leaves the
converter's output a pure function of the dataset.

``PUT`` is still used for the two things that have no update handler --
creating a database and installing a design document.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import io
import ssl
import json
import time
import base64
import netrc
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["CouchDB", "CouchError", "design_from_dir"]


class CouchError(RuntimeError):
    """A CouchDB request failed.  ``status`` and ``body`` carry the response."""

    def __init__(self, status, body, url=""):
        self.status = status
        self.body = body
        self.url = url
        super().__init__("CouchDB %s on %s: %s" % (status, url, body))


class CouchDB:
    def __init__(
        self,
        url=None,
        user=None,
        password=None,
        netrc_machine=None,
        timeout=600,
        retries=4,
        verify=True,
        retry_body_limit=1 << 20,
    ):
        self.url = (url or os.environ.get("NEUROJSON_IO") or "http://localhost:5984").rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_body_limit = int(retry_body_limit)
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.username and not user:
            user, password = parsed.username, parsed.password
            # strip credentials out of the stored URL so they cannot leak into
            # logs or error messages
            netloc = parsed.hostname + (":%d" % parsed.port if parsed.port else "")
            self.url = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", "")).rstrip(
                "/"
            )
        if user is None and netrc_machine:
            try:
                auth = netrc.netrc().hosts.get(netrc_machine)
                if auth:
                    user, password = auth[0], auth[2]
            except (OSError, netrc.NetrcParseError):
                pass
        self.user = user
        self.password = password
        self._ctx = None
        if not verify:
            self._ctx = ssl.create_default_context()
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE

    # -- transport ----------------------------------------------------------

    def request(self, method, path, body=None, raw=None, ctype="application/json"):
        """Issue one request; returns ``(status, parsed_body)``.

        5xx and 409 are retried with exponential backoff -- a bulk publish of a
        few thousand documents will otherwise trip over transient view-builder
        load on the server.
        """
        url = self.url + "/" + path.lstrip("/")
        data = (
            raw
            if raw is not None
            else (json.dumps(body).encode("utf-8") if body is not None else None)
        )
        delay = 1.0
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, method=method, data=data)
            req.add_header("Accept", "application/json")
            if data is not None:
                req.add_header("Content-Type", ctype)
            if self.user is not None:
                token = base64.b64encode(
                    ("%s:%s" % (self.user, self.password or "")).encode()
                ).decode()
                req.add_header("Authorization", "Basic " + token)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as res:
                    payload = res.read()
                    return res.status, (json.loads(payload) if payload else {})
            except urllib.error.HTTPError as err:
                payload = err.read()
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    parsed = {"raw": payload[:500].decode("utf-8", "replace")}
                if err.code >= 500 or err.code == 409:
                    last = (err.code, parsed)
                    if attempt < self.retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                return err.code, parsed
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last = (0, {"error": str(err)})
                # A large body that gets its connection reset is far more
                # likely to have been rejected for size than to have hit a
                # transient fault: CouchDB answers an oversized PUT with a
                # clean 413, but an oversized POST to an update handler by
                # closing the socket.  Re-uploading many megabytes four times
                # to rediscover that is pure waste, so transport failures are
                # only retried for small requests.
                retryable = attempt < self.retries and (
                    data is None or len(data) <= self.retry_body_limit
                )
                if retryable:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CouchError(0, str(err), url)
        return last if last else (0, {"error": "unreachable"})

    def _ok(self, method, path, body=None, expect=(200, 201, 202)):
        status, payload = self.request(method, path, body=body)
        if status not in expect:
            raise CouchError(status, payload, path)
        return payload

    # -- server / database --------------------------------------------------

    def welcome(self):
        return self._ok("GET", "/")

    def session(self):
        return self._ok("GET", "/_session")

    def is_server_admin(self):
        status, _payload = self.request("GET", "/_membership")
        return status == 200

    def all_dbs(self):
        return self._ok("GET", "/_all_dbs")

    def db_exists(self, db):
        status, _ = self.request("GET", "/" + urllib.parse.quote(db, safe=""))
        return status == 200

    def db_info(self, db):
        return self._ok("GET", "/" + urllib.parse.quote(db, safe=""))

    def create_db(self, db, exist_ok=True):
        """Create a database (requires server-admin rights)."""
        status, payload = self.request("PUT", "/" + urllib.parse.quote(db, safe=""))
        if status in (201, 202):
            return payload
        if status == 412 and exist_ok:
            return {"ok": True, "existed": True}
        raise CouchError(status, payload, db)

    def delete_db(self, db):
        return self._ok("DELETE", "/" + urllib.parse.quote(db, safe=""))

    def set_security(self, db, admins=None, members=None):
        body = {
            "admins": {"names": list(admins or []), "roles": []},
            "members": {"names": list(members or []), "roles": []},
        }
        return self._ok("PUT", "/%s/_security" % urllib.parse.quote(db, safe=""), body)

    def get_security(self, db):
        return self._ok("GET", "/%s/_security" % urllib.parse.quote(db, safe=""))

    # -- documents ----------------------------------------------------------

    def get_doc(self, db, docid, **params):
        path = "/%s/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(docid, safe=""),
        )
        if params:
            path += "?" + urllib.parse.urlencode(params)
        return self._ok("GET", path)

    def doc_exists(self, db, docid):
        status, _ = self.request(
            "GET",
            "/%s/%s" % (urllib.parse.quote(db, safe=""), urllib.parse.quote(docid, safe="")),
        )
        return status == 200

    def push(self, db, docid, doc, design="qq", handler="timestamp"):
        """Publish a data document through the update handler.

        Deliberately a POST: the handler owns ``_rev`` resolution and the
        timestamp metadata, so the caller neither reads the current revision nor
        writes any clock value of its own.
        """
        path = "/%s/_design/%s/_update/%s/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(design, safe=""),
            urllib.parse.quote(handler, safe=""),
            urllib.parse.quote(docid, safe=""),
        )
        status, payload = self.request("POST", path, body=doc)
        if status not in (200, 201, 202):
            raise CouchError(status, payload, path)
        return payload

    def push_file(self, db, docid, jsonfile, design="qq", handler="timestamp"):
        """Publish a document straight from a file, without parsing it.

        Avoids materialising a multi-megabyte document as Python objects purely
        to re-serialise it, and guarantees the published bytes are exactly the
        canonical bytes that were fingerprinted.
        """
        path = "/%s/_design/%s/_update/%s/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(design, safe=""),
            urllib.parse.quote(handler, safe=""),
            urllib.parse.quote(docid, safe=""),
        )
        with open(jsonfile, "rb") as fid:
            raw = fid.read()
        status, payload = self.request("POST", path, raw=raw)
        if status not in (200, 201, 202):
            raise CouchError(status, payload, path)
        return payload

    def put_doc(self, db, docid, doc, rev=None):
        """Write a document with a plain PUT, preserving ``_rev``.

        Dataset digests must never take this path -- they go through the update
        handler, which owns ``_rev`` resolution and the timestamp metadata.  This
        exists for *configuration* documents in databases that have no such
        handler, the ``sys/registry`` entry being the case that matters: it is a
        shared document listing every database the search layer should index, so
        it has to be read, amended and written back at the revision it was read
        at, or a concurrent edit is silently discarded.
        """
        path = "/%s/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(docid, safe=""),
        )
        body = dict(doc)
        if rev is None:
            rev = body.get("_rev")
        if rev is None:
            status, current = self.request("GET", path)
            if status == 200:
                rev = current.get("_rev")
        if rev:
            body["_rev"] = rev
        else:
            body.pop("_rev", None)
        status, payload = self.request("PUT", path, body=body)
        if status not in (200, 201, 202):
            raise CouchError(status, payload, path)
        return payload

    def delete_doc(self, db, docid, rev=None):
        if rev is None:
            rev = self.get_doc(db, docid)["_rev"]
        return self._ok(
            "DELETE",
            "/%s/%s?rev=%s"
            % (
                urllib.parse.quote(db, safe=""),
                urllib.parse.quote(docid, safe=""),
                urllib.parse.quote(rev, safe=""),
            ),
        )

    def all_docs(self, db, **params):
        path = "/%s/_all_docs" % urllib.parse.quote(db, safe="")
        if params:
            path += "?" + urllib.parse.urlencode(params)
        return self._ok("GET", path)

    # -- design documents / views -------------------------------------------

    def put_design(self, db, ddoc, name="qq"):
        """Install or update a design document.

        A design document has no update handler of its own, so this is the one
        place a ``PUT`` is correct; the current ``_rev`` is fetched first so an
        existing design document is updated rather than rejected.
        """
        path = "/%s/_design/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(name, safe=""),
        )
        body = dict(ddoc)
        body.pop("_rev", None)
        body["_id"] = "_design/%s" % name
        status, current = self.request("GET", path)
        if status == 200 and "_rev" in current:
            body["_rev"] = current["_rev"]
        status, payload = self.request("PUT", path, body=body)
        if status not in (200, 201, 202):
            raise CouchError(status, payload, path)
        return payload

    def get_design(self, db, name="qq"):
        return self._ok(
            "GET",
            "/%s/_design/%s" % (urllib.parse.quote(db, safe=""), urllib.parse.quote(name, safe="")),
        )

    def view(self, db, view, design="qq", **params):
        path = "/%s/_design/%s/_view/%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.quote(design, safe=""),
            urllib.parse.quote(view, safe=""),
        )
        query = {}
        for key, val in params.items():
            query[key] = val if isinstance(val, str) else json.dumps(val)
        if query:
            path += "?" + urllib.parse.urlencode(query)
        return self._ok("GET", path)

    def warm_view(self, db, view, design="qq"):
        """Force a view index build and report how long it took."""
        start = time.time()
        res = self.view(db, view, design=design, limit=1)
        return {"view": view, "seconds": time.time() - start, "total": res.get("total_rows")}

    def alive(self):
        """True if the server answers a trivial request.

        Used to tell a size rejection apart from a real network fault: an
        oversized POST to an update handler arrives as a bare connection reset,
        which on its own is indistinguishable from the server having gone away.
        """
        try:
            status, _body = self.request("GET", "/_up")
            return status == 200
        except CouchError:
            return False

    def changes(self, db, since="0", limit=None):
        params = {"since": since}
        if limit:
            params["limit"] = limit
        path = "/%s/_changes?%s" % (
            urllib.parse.quote(db, safe=""),
            urllib.parse.urlencode(params),
        )
        return self._ok("GET", path)


def design_from_dir(path, language="javascript"):
    """Assemble a design document from a directory of ``.js`` files.

    ==============================  =========================================
    ``view_<name>.js``              map function of view ``<name>``
    ``reduce_<name>.js``            reduce function of view ``<name>``
    ``update_<name>.js``            update handler ``<name>``
    ``filter_<name>.js``            filter function ``<name>``
    ``validate_doc_update.js``      write-validation function
    ==============================  =========================================

    Keeping the JavaScript in reviewable, diffable files rather than embedded in
    a JSON blob is what makes a view change auditable -- the design document
    that was live before this work existed only on the server, in no repository.
    """
    ddoc = {"language": language, "views": {}, "updates": {}, "filters": {}}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".js"):
            continue
        with open(os.path.join(path, name), "r", encoding="utf-8") as fid:
            code = fid.read()
        stem = name[: -len(".js")]
        if stem == "validate_doc_update":
            ddoc["validate_doc_update"] = code
        elif stem.startswith("view_"):
            ddoc["views"].setdefault(stem[5:], {})["map"] = code
        elif stem.startswith("reduce_"):
            ddoc["views"].setdefault(stem[7:], {})["reduce"] = code
        elif stem.startswith("update_"):
            ddoc["updates"][stem[7:]] = code
        elif stem.startswith("filter_"):
            ddoc["filters"][stem[7:]] = code
    for key in ("updates", "filters"):
        if not ddoc[key]:
            del ddoc[key]
    return ddoc
