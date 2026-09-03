"""
Command line pipeline for publishing dataset digests to NeuroJSON.

    python3 -m jdata.njcli convert --input DIR --output DIR --db NAME [--ds ...]
    python3 -m jdata.njcli push    --output DIR --db NAME --server URL [--ds ...]
    python3 -m jdata.njcli views   --db NAME --design DIR --server URL
    python3 -m jdata.njcli cas     --cas DIR {usage,verify}
    python3 -m jdata.njcli report  --output DIR

The stages are separate commands on purpose.  Conversion is CPU/IO bound and
runs where the data is; publishing is a network operation against a database
that may be unreachable or read-only at the time.  Keeping them apart means a
failed publish never forces a re-conversion, and the immutable per-version
output on disk stays the source of truth.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import sys
import json
import time
import re
import argparse
import traceback
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed

from .njcas import CAS
from .njbids import NJBIDS_DEFAULT, bids2json, canonical_json

__all__ = ["main", "convert_dataset", "dataset_outdir"]


# =============================================================================
# output layout
# =============================================================================


def dataset_outdir(outputroot, dsname, version):
    return os.path.join(outputroot, dsname, version)


def _write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fid:
        fid.write(text)
    os.replace(tmp, path)


def _relink(link, target):
    tmp = "%s.tmp.%d" % (link, os.getpid())
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, link)


def convert_dataset(dspath, outputroot, dbname, casroot, options, force=False):
    """Convert one dataset and write its immutable per-version output.

    Returns a small summary dict; nothing large crosses the process boundary so
    this is safe to fan out with a process pool.
    """
    dsname = os.path.basename(dspath.rstrip("/"))
    started = time.time()
    cas = CAS(
        casroot,
        mode=options.get("cas_mode", "link"),
        algo=options.get("cas_algo", "sha256"),
        annex_hash=options.get("cas_annex_hash", False),
    )
    try:
        result = bids2json(
            dspath, dbname=dbname, dsname=dsname, cas=cas, **options.get("njbids", {})
        )
    finally:
        cas.flush()
        cas.close()

    version = result["version"]["Version"]
    outdir = dataset_outdir(outputroot, dsname, version)
    docpath = os.path.join(outdir, "doc.json")

    doctext = canonical_json(result["doc"])
    if (not force) and os.path.exists(docpath):
        with open(docpath, "r", encoding="utf-8") as fid:
            if fid.read() == doctext:
                return {
                    "ds": dsname,
                    "version": version,
                    "fingerprint": result["fingerprint"],
                    "status": "unchanged",
                    "docbytes": len(doctext),
                    "files": len(result["manifest"]),
                    "seconds": time.time() - started,
                    "errors": len(result["errors"]),
                }

    _write_atomic(docpath, doctext)
    _write_atomic(os.path.join(outdir, "manifest.tsv"), result["manifest_blob"])
    for name, subdoc in result["split"].items():
        _write_atomic(os.path.join(outdir, "%s.json" % name), canonical_json(subdoc))
    _write_atomic(
        os.path.join(outdir, "meta.json"),
        json.dumps(
            {
                "dataset": dsname,
                "database": dbname,
                "version": result["version"],
                "fingerprint": result["fingerprint"],
                "stats": result["stats"],
                "errors": result["errors"],
                "docbytes": len(doctext),
                "split": {k: len(canonical_json(v)) for k, v in result["split"].items()},
            },
            indent=1,
            sort_keys=True,
        ),
    )
    _relink(os.path.join(outputroot, dsname, "latest"), version)

    return {
        "ds": dsname,
        "version": version,
        "fingerprint": result["fingerprint"],
        "status": "written",
        "docbytes": len(doctext),
        "files": len(result["manifest"]),
        "seconds": time.time() - started,
        "errors": len(result["errors"]),
        "offloaded": len(result["stats"].get("offloaded", [])),
    }


def _convert_worker(args):
    dspath, outputroot, dbname, casroot, options, force = args
    try:
        return convert_dataset(dspath, outputroot, dbname, casroot, options, force)
    except Exception as err:
        return {
            "ds": os.path.basename(dspath.rstrip("/")),
            "status": "failed",
            "error": "%s: %s" % (type(err).__name__, err),
            "traceback": traceback.format_exc(limit=6),
        }


# =============================================================================
# commands
# =============================================================================


def _dataset_paths(inputroot, names):
    if names:
        return [os.path.join(inputroot, n) for n in names]
    return sorted(
        os.path.join(inputroot, n)
        for n in os.listdir(inputroot)
        if os.path.isdir(os.path.join(inputroot, n)) and not n.startswith(".")
    )


def cmd_convert(args):
    paths = _dataset_paths(args.input, args.ds)
    paths = [p for p in paths if os.path.isdir(p)]
    if not paths:
        print("no datasets found under %s" % args.input, file=sys.stderr)
        return 1

    # "annex" takes the content hash out of the git-annex key instead of reading
    # the payload.  OpenNeuro's annex backend is MD5E, so the key already states
    # both the content hash and the exact size, and hardlinking is a metadata
    # operation -- the whole pass then needs no payload reads at all.  Choosing
    # sha256 instead means reading every byte of the mirror once.
    # Encoding does not change how *unencoded* files are identified.  An
    # attachment is named by the sha256 of its source, which the encoder
    # computes while it reads the payload it is about to re-encode anyway; but
    # promoting every other file to sha256 as well would mean reading the whole
    # mirror to hash things that are only ever referenced -- tens of thousands
    # of JPEGs per dataset, in the worst case observed.  Those keep the hash the
    # annex key already carries, and the URL states its algorithm, so a document
    # holding both is self-describing.
    content_hash = args.content_hash
    algo = "md5" if content_hash == "annex" else "sha256"
    # annex-derived hashes stay enabled even for a sha256 store: a SHA256E key
    # then supplies the digest for free, and only MD5E-keyed files are read.
    annex_hash = True
    options = {
        "cas_mode": args.cas_mode,
        "cas_algo": algo,
        "cas_annex_hash": annex_hash,
        "njbids": {"hash_algorithm": algo, "hash_source": content_hash},
    }
    if args.max_doc:
        options["njbids"]["max_doc"] = args.max_doc
    if args.cas_url:
        options["njbids"]["cas_url"] = args.cas_url
    if args.file_threads:
        options["njbids"]["hash_threads"] = args.file_threads
    if args.encode:
        options["njbids"]["encode"] = tuple(args.encode)
        options["njbids"]["encode_codec"] = args.encode_codec
        options["njbids"]["encode_threads"] = args.encode_threads
    if args.max_encode:
        options["njbids"]["max_encode"] = args.max_encode
    if args.no_split:
        options["njbids"]["split_dirs"] = ()

    casroot = args.cas or os.environ.get("NEUROJSON_CAS_ROOT")
    if not casroot:
        print("--cas or $NEUROJSON_CAS_ROOT is required", file=sys.stderr)
        return 1

    jobs = [(p, args.output, args.db, casroot, options, args.force) for p in paths]
    started = time.time()
    summary = {"written": 0, "unchanged": 0, "failed": 0}
    results = []
    logfid = open(args.log, "a", encoding="utf-8") if args.log else None

    def record(res):
        results.append(res)
        summary[res.get("status", "failed")] = summary.get(res.get("status"), 0) + 1
        if logfid:
            logfid.write(json.dumps(res) + "\n")
            logfid.flush()
        done = len(results)
        if res.get("status") == "failed":
            print("  [%d/%d] %-12s FAILED %s" % (done, len(jobs), res["ds"], res.get("error")))
        elif args.verbose or done % 25 == 0 or done == len(jobs):
            print(
                "  [%d/%d] %-12s %-9s %-14s %8.1f kB %5d files %6.1fs%s"
                % (
                    done,
                    len(jobs),
                    res["ds"],
                    res["status"],
                    res.get("version", "-"),
                    res.get("docbytes", 0) / 1024.0,
                    res.get("files", 0),
                    res.get("seconds", 0),
                    "  offload=%d" % res["offloaded"] if res.get("offloaded") else "",
                )
            )

    print("converting %d dataset(s) with %d worker(s)" % (len(jobs), args.threads))
    if args.threads > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(_convert_worker, job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    else:
        for job in jobs:
            record(_convert_worker(job))

    if logfid:
        logfid.close()
    elapsed = time.time() - started
    bytes_out = sum(r.get("docbytes", 0) for r in results)
    print(
        "\n%d written, %d unchanged, %d failed in %.1fs (%.1f ds/min, %.1f MB of JSON)"
        % (
            summary.get("written", 0),
            summary.get("unchanged", 0),
            summary.get("failed", 0),
            elapsed,
            60.0 * len(jobs) / elapsed if elapsed else 0,
            bytes_out / 1e6,
        )
    )
    bad = [r for r in results if r.get("status") == "failed"]
    for res in bad[:10]:
        print("FAILED %s: %s" % (res["ds"], res.get("error")))
    return 1 if bad else 0


#: files in a version directory that are not documents to publish
NON_DOCUMENT_FILES = ("doc.json", "meta.json", "datacite.json")


def _iter_published(outputroot, names=None, split_names=None):
    """Yield ``(dsname, version, docpath, splitpaths)`` for each latest version.

    Split documents are recognised by name against the known split directories,
    not by "any .json that is not doc.json".  The loose test previously picked up
    the DataCite record written alongside the document and tried to publish it as
    a derivatives document.
    """
    for dsname in sorted(names or os.listdir(outputroot)):
        dsdir = os.path.join(outputroot, dsname)
        latest = os.path.join(dsdir, "latest")
        if not os.path.isdir(dsdir) or not os.path.islink(latest):
            continue
        version = os.readlink(latest)
        vdir = os.path.join(dsdir, version)
        docpath = os.path.join(vdir, "doc.json")
        if not os.path.isfile(docpath):
            continue
        known = tuple(split_names or NJBIDS_DEFAULT["split_dirs"])
        splits = {
            name[: -len(".json")]: os.path.join(vdir, name)
            for name in sorted(os.listdir(vdir))
            if name.endswith(".json")
            and name not in NON_DOCUMENT_FILES
            and name[: -len(".json")] in known
        }
        yield dsname, version, docpath, splits


def _trim_largest_subtree(docpath, cas, db, ds, cas_url_base=None):
    """Shed the largest top-level subtree of a document into the store.

    Returns a description of what was shed, or None if nothing remains that may
    be shed.  The document on disk is rewritten, so the immutable archive keeps
    matching what was actually published.

    Never sheds the dataset-level metadata that makes a document findable --
    description, participants table, README, the metadata block -- and always
    takes the largest remaining subtree, so the result is a function of the
    document rather than of how many attempts it took to get there.
    """
    from .njbids import BUDGET_PROTECTED, canonical_json
    from .njcas import cas_url

    with open(docpath, "r", encoding="utf-8") as fid:
        doc = json.load(fid)

    candidates = []
    for key, value in doc.items():
        if key in BUDGET_PROTECTED or not isinstance(value, dict):
            continue
        if "_DataLink_" in value:
            continue
        candidates.append((len(canonical_json(value)), key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    was, key = candidates[0]

    digest, size = cas.put_bytes(canonical_json(doc[key]))
    doc[key] = {
        "_DataLink_": cas_url(
            digest,
            size=size,
            db=db,
            doc=ds,
            file="%s.json" % key,
            base=cas_url_base,
            algo=cas.algo,
        )
    }
    _write_atomic(docpath, canonical_json(doc))
    return {"path": key, "was": was, "now": len(canonical_json(doc))}


def cmd_push(args):
    """Publish digests, trimming any the server rejects as too large.

    A JSON byte count is the wrong thing to budget against: CouchDB limits the
    *internal* size of a parsed document, and the ratio to JSON depends entirely
    on content.  Measured against CouchDB 3.4.2 with an 8 MB limit, the largest
    JSON accepted was 4.19 MB for one big string, 7.23 MB for many short keys,
    and over 29 MB for a float array.  Rather than guess at that from the
    client, documents are converted whole and trimmed here, once the server has
    actually said no.
    """
    from .njcouch import CouchDB, CouchError

    if not _guard_production(args.server, args.allow_production):
        return 1
    couch = CouchDB(args.server, netrc_machine=args.netrc)
    cas = CAS(args.cas, algo=args.algo) if args.cas else None

    items = list(_iter_published(args.output, args.ds))
    print("pushing %d document(s) to %s/%s" % (len(items), couch.url, args.db))
    ok = failed = trimmed = 0
    for dsname, version, docpath, splits in items:
        notes = []
        published = False
        for attempt in range(args.max_trim + 1):
            try:
                couch.push_file(args.db, dsname, docpath, design=args.design)
                published = True
                break
            except CouchError as err:
                # 413/document_too_large is the clean answer a PUT gets; a POST
                # to an update handler is answered by closing the socket
                # instead, so a transport failure counts as too-large provided
                # the server is still there.
                toolarge = "too_large" in str(err.body) or err.status == 413
                if not toolarge and err.status == 0 and couch.alive():
                    toolarge = True
                if not toolarge:
                    print("  %-12s FAILED %s %s" % (dsname, err.status, err.body))
                    break
                if cas is None:
                    print("  %-12s too large, and no --cas given to trim into" % dsname)
                    break
                if attempt >= args.max_trim:
                    print("  %-12s still too large after %d trims" % (dsname, args.max_trim))
                    break
                shed = _trim_largest_subtree(
                    docpath, cas, args.db, dsname, cas_url_base=args.cas_url
                )
                if shed is None:
                    print("  %-12s too large, nothing left to trim" % dsname)
                    break
                trimmed += 1
                notes.append("%s %.0fkB" % (shed["path"], shed["was"] / 1024.0))

        if not published:
            failed += 1
            continue

        for name, path in splits.items():
            target = args.split_db or ("%s_%s" % (args.db, name.rstrip("s")))
            try:
                couch.push_file(target, dsname, path, design=args.design)
            except CouchError as err:
                print("  %-12s %s push failed %s %s" % (dsname, name, err.status, err.body))
        ok += 1
        if args.verbose or notes:
            print(
                "  %-12s %-14s %8.1f kB%s"
                % (
                    dsname,
                    version,
                    os.path.getsize(docpath) / 1024.0,
                    ("  trimmed: " + ", ".join(notes)) if notes else "",
                )
            )
    if cas:
        cas.close()
    print("%d pushed, %d failed, %d subtree(s) trimmed to fit" % (ok, failed, trimmed))
    return 1 if failed else 0


PRODUCTION_HOSTS = ("neurojson.io", "neurojson.org")


def _guard_production(server, allowed):
    """Refuse to write to the production host unless explicitly permitted.

    The admin credential for production is routinely present in the operator's
    environment, so an accidentally copied URL is a realistic way to write to
    the live database.  A publish is not trivially reversible, so the default
    has to be refusal.
    """
    host = urllib.parse.urlsplit(server or "").hostname or ""
    if any(host == h or host.endswith("." + h) for h in PRODUCTION_HOSTS) and not allowed:
        print(
            "refusing to write to production host %r without --allow-production" % host,
            file=sys.stderr,
        )
        return False
    return True


def cmd_deploy(args):
    """Create the databases, install the design document and register them."""
    from .njcouch import CouchDB, CouchError, design_from_dir

    if not _guard_production(args.server, args.allow_production):
        return 1
    couch = CouchDB(args.server, netrc_machine=args.netrc)

    session = couch.session().get("userCtx", {})
    print("server %s as %s (roles=%s)" % (couch.url, session.get("name"), session.get("roles")))

    databases = [args.db] + ([args.split_db] if args.split_db else [])
    missing = [db for db in databases if not couch.db_exists(db)]

    # Server-admin rights are only needed to *create* a database.  Installing a
    # design document and publishing documents need database-admin rights,
    # which is a much smaller grant, so do not demand more than the run needs.
    if missing and not couch.is_server_admin():
        print(
            "\nERROR: %s do(es) not exist, and this account (%s) is not a CouchDB\n"
            "server admin, so it cannot create databases. Either create them with a\n"
            "server-admin credential, or ask an administrator for:\n"
            "    curl -X PUT http://<host>/%s\n"
            "    curl -X PUT http://<host>/%s/_security \\\n"
            '         -d \'{"admins":{"names":["%s"],"roles":[]},"members":{"names":[],"roles":[]}}\''
            % (
                ", ".join(missing),
                session.get("name"),
                missing[0],
                missing[0],
                session.get("name"),
            ),
            file=sys.stderr,
        )
        return 2

    for db in databases:
        if db in missing:
            couch.create_db(db)
            print("  %-24s created" % db)
        else:
            info = couch.db_info(db)
            print("  %-24s exists (%d docs)" % (db, info["doc_count"]))
        if args.admin:
            try:
                couch.set_security(db, admins=args.admin, members=args.member or [])
                print("  %-24s security: admins=%s" % (db, args.admin))
            except CouchError as err:
                print("  %-24s security unchanged (%s)" % (db, err.status))

    ddoc = design_from_dir(args.design)
    for db in databases:
        couch.put_design(db, ddoc, name=args.name)
        print(
            "  %-24s _design/%s installed (views=%s)"
            % (db, args.name, sorted(ddoc.get("views", {})))
        )

    if args.register:
        _register(couch, args, databases)

    if args.warm:
        for db in databases:
            for view in sorted(ddoc.get("views", {})):
                info = couch.warm_view(db, view, design=args.name)
                print(
                    "  %-24s %-18s %7.1fs rows=%s"
                    % (db, info["view"], info["seconds"], info["total"])
                )
    return 0


def _register(couch, args, databases):
    """Add the databases to the sys/registry document.

    The Postgres sync discovers which databases to index by reading
    sys/registry, so a database absent from it is never searchable no matter how
    many documents it holds.

    The registry is a shared configuration document in a database with no update
    handler, so it is read, amended and written back at the revision it was read
    at -- rather than through the timestamp handler that dataset digests use.
    """
    from .njcouch import CouchError

    try:
        registry = couch.get_doc("sys", "registry")
    except CouchError as err:
        print("  registry unavailable (%s); skipping registration" % err.status)
        return
    entries = registry.get("database", [])
    known = {entry.get("id") for entry in entries}
    added = []
    for db in databases:
        if db in known:
            continue
        entry = {
            "id": db,
            "name": args.register_name or db,
            "fullname": args.register_name or db,
            "url": args.register_url or "",
            "group": 1,
            "datatype": list(args.register_datatype or []),
            "standard": ["BIDS"],
        }
        entries.append(entry)
        added.append(db)
    if not added:
        print("  registry already lists %s" % ", ".join(databases))
        return
    registry["database"] = entries
    try:
        couch.put_doc("sys", "registry", registry)
    except CouchError as err:
        print("  registry NOT updated (%s: %s)" % (err.status, err.body))
        return
    print("  registry updated: added %s (now %d databases)" % (", ".join(added), len(entries)))


def cmd_views(args):
    from .njcouch import CouchDB, design_from_dir

    if not _guard_production(args.server, getattr(args, "allow_production", False)):
        return 1
    couch = CouchDB(args.server, netrc_machine=args.netrc)
    ddoc = design_from_dir(args.design)
    print(
        "installing _design/%s in %s: views=%s updates=%s"
        % (args.name, args.db, sorted(ddoc.get("views", {})), sorted(ddoc.get("updates", {})))
    )
    couch.put_design(args.db, ddoc, name=args.name)
    if args.warm:
        for view in sorted(ddoc.get("views", {})):
            info = couch.warm_view(args.db, view, design=args.name)
            print("  %-20s %8.1fs  total_rows=%s" % (info["view"], info["seconds"], info["total"]))
    return 0


def cmd_cas(args):
    cas = CAS(args.cas)
    if args.action == "usage":
        usage = cas.usage()
        print(json.dumps({**usage, "memo_rows": cas.memo_count()}, indent=1))
        if usage["objects"]:
            print(
                "shared-inode fraction: %.4f (1.0 means no extra space used)"
                % (1.0 - usage["exclusive_bytes"] / max(1, usage["apparent_bytes"]))
            )
    else:
        report = cas.verify(sample=args.sample)
        print(json.dumps(report, indent=1))
        if report["bad"]:
            return 1
    cas.close()
    return 0


_LINK_HASH = re.compile(r"hash=([a-z0-9]+):([0-9a-fA-F]+)")


def _iter_links(node, where="$"):
    """Yield ``(algo, digest, jsonpath)`` for every _DataLink_ in a document."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_DataLink_" and isinstance(value, str):
                match = _LINK_HASH.search(value)
                if match:
                    yield match.group(1), match.group(2).lower(), where
            else:
                yield from _iter_links(value, "%s.%s" % (where, key.replace(".", "\\.")))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_links(value, "%s[%d]" % (where, index))


def _read_manifest(path):
    entries = []
    if not os.path.isfile(path):
        return entries
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and not parts[0].startswith("payload"):
                algo, _sep, digest = parts[0].partition(":")
                entries.append(
                    {
                        "algo": algo,
                        "sha256": digest if digest not in ("", "None") else None,
                        "size": int(parts[1]) if parts[1] not in ("", "None") else None,
                        "path": parts[2],
                    }
                )
    return entries


def cmd_verify(args):
    """Reconcile the on-disk archive, its manifest, and the content store.

    Three independent checks, because a DOI is a claim about bytes and each
    check can fail on its own:

    * the fingerprint recorded in the document must equal the fingerprint
      recomputed from the document and its manifest -- proving the archive has
      not been edited since it was written;
    * every manifest entry must name an object that is actually present in the
      store -- proving the links resolve;
    * with --deep, a sample of those objects is re-hashed -- proving the bytes
      themselves are intact.
    """
    from .njbids import fingerprint, _dehydrate

    cas = (
        CAS(args.cas, mode="none", algo=args.algo, annex_hash=(args.algo != "sha256"))
        if args.cas
        else None
    )
    checked = fp_bad = missing = 0
    problems = []

    for dsname, version, docpath, _splits in _iter_published(args.output, args.ds):
        vdir = os.path.dirname(docpath)
        with open(docpath, "r", encoding="utf-8") as fid:
            doc = json.load(fid)
        recorded = (doc.get(".neurojson") or {}).get("Fingerprint")
        manifest = _read_manifest(os.path.join(vdir, "manifest.tsv"))

        # the fingerprint is computed over the document *without* the metadata
        # block, since the block carries the fingerprint itself
        payload = {k: v for k, v in doc.items() if k != ".neurojson"}
        recomputed, _blob = fingerprint(payload, manifest)
        checked += 1
        if recorded and recomputed != recorded:
            fp_bad += 1
            problems.append("%s@%s: fingerprint mismatch" % (dsname, version))

        if cas:
            # Check the links, not the manifest.  Every file is manifested,
            # including the ones whose content is inlined in the document and
            # therefore deliberately never materialised as an object; only a
            # _DataLink_ makes a promise that something is retrievable.
            absent, unfetched = [], 0
            for algo, digest, where in _iter_links(doc):
                if algo != cas.algo:
                    # a link under a different algorithm than this store uses
                    # (e.g. an annex-key reference to content never fetched)
                    unfetched += 1
                    continue
                if not cas.has(digest):
                    absent.append(where)
            if absent:
                missing += len(absent)
                problems.append(
                    "%s@%s: %d link(s) do not resolve in the store, e.g. %s"
                    % (dsname, version, len(absent), absent[0])
                )
            if unfetched and args.verbose:
                print("    %d link(s) reference content not fetched locally" % unfetched)
        if args.verbose:
            print(
                "  %-12s %-14s %5d files %s"
                % (dsname, version, len(manifest), "ok" if not problems else "see below")
            )

    print(
        "%d version(s) checked: %d fingerprint mismatch, %d manifest object(s) missing"
        % (checked, fp_bad, missing)
    )
    for line in problems[:20]:
        print("  " + line)

    if cas and args.deep:
        report = cas.verify(sample=args.sample)
        print(
            "store: %d object(s) re-hashed, %d ok, %d corrupt, %d hardlinked"
            % (report["checked"], report["ok"], len(report["bad"]), report["hardlinks"])
        )
        for bad in report["bad"][:10]:
            print("  corrupt object %s (actual %s)" % (bad["object"][:16], bad["actual"][:16]))
        if report["bad"]:
            problems.append("corrupt store objects")
    if cas:
        cas.close()
    return 1 if problems else 0


def cmd_doi(args):
    """Emit a DataCite record per dataset version, next to the version archive."""
    from .njdoi import datacite

    written = 0
    for dsname, version, docpath, _splits in _iter_published(args.output, args.ds):
        vdir = os.path.dirname(docpath)
        with open(docpath, "r", encoding="utf-8") as fid:
            doc = json.load(fid)
        manifest = []
        mpath = os.path.join(vdir, "manifest.tsv")
        if os.path.isfile(mpath):
            with open(mpath, "r", encoding="utf-8") as fid:
                for line in fid:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        manifest.append(
                            {"sha256": parts[0], "size": int(parts[1] or 0), "path": parts[2]}
                        )
        record = datacite(
            doc,
            args.db,
            dsname,
            manifest=manifest,
            publisher=args.publisher,
            landing_base=args.landing_base,
            publication_year=args.year,
        )
        text = json.dumps(record, indent=1, sort_keys=True)
        if args.stdout:
            print(text)
        else:
            _write_atomic(os.path.join(vdir, "datacite.json"), text)
            written += 1
            if args.verbose:
                print("  %-12s %-14s %s" % (dsname, version, record.get("titles")[0]["title"][:50]))
    if not args.stdout:
        print("wrote %d datacite.json record(s)" % written)
    return 0


def cmd_report(args):
    rows = []
    for dsname, version, docpath, splits in _iter_published(args.output, args.ds):
        metapath = os.path.join(os.path.dirname(docpath), "meta.json")
        meta = {}
        if os.path.isfile(metapath):
            with open(metapath, "r", encoding="utf-8") as fid:
                meta = json.load(fid)
        rows.append(
            {
                "ds": dsname,
                "version": version,
                "docbytes": os.path.getsize(docpath),
                "files": meta.get("stats", {}).get("files"),
                "errors": len(meta.get("errors", [])),
                "offloaded": len(meta.get("stats", {}).get("offloaded", [])),
                "fingerprint": meta.get("fingerprint", "")[:16],
            }
        )
    rows.sort(key=lambda r: -r["docbytes"])
    total = sum(r["docbytes"] for r in rows)
    over = [r for r in rows if r["docbytes"] > args.max_doc]
    print(
        "%d datasets, %.1f MB of JSON, %d over %d bytes"
        % (len(rows), total / 1e6, len(over), args.max_doc)
    )
    print(
        "%-12s %-14s %10s %7s %7s %6s  %s"
        % ("dataset", "version", "docbytes", "files", "errors", "offl", "fingerprint")
    )
    for row in rows[: args.top]:
        print(
            "%-12s %-14s %10d %7s %7d %6d  %s"
            % (
                row["ds"],
                row["version"],
                row["docbytes"],
                row["files"],
                row["errors"],
                row["offloaded"],
                row["fingerprint"],
            )
        )
    return 0


# =============================================================================
# argument parsing
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(prog="jdata.njcli", description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert dataset folders to JSON digests")
    conv.add_argument("--input", required=True, help="root holding the dataset folders")
    conv.add_argument("--output", required=True, help="output root for per-version digests")
    conv.add_argument("--db", required=True, help="database name used in _DataLink_ URLs")
    conv.add_argument("--ds", nargs="*", help="dataset names (default: all)")
    conv.add_argument("--cas", help="content store root (or $NEUROJSON_CAS_ROOT)")
    conv.add_argument("--cas-mode", default="link", choices=["link", "symlink", "copy", "none"])
    conv.add_argument("--cas-url", help="base URL template for _DataLink_")
    conv.add_argument(
        "--content-hash",
        default="annex",
        choices=["annex", "sha256"],
        help="how attachment identifiers are derived. 'annex' (default) reads the "
        "content hash out of the git-annex key, which costs no payload I/O; "
        "'sha256' hashes every payload, which means reading the whole mirror once",
    )
    conv.add_argument("--threads", type=int, default=8, help="parallel datasets")
    conv.add_argument(
        "--file-threads",
        type=int,
        help="threads used to pre-hash payloads within one dataset; useful when "
        "converting a handful of very large datasets, where per-dataset "
        "parallelism leaves most of the pool idle",
    )
    conv.add_argument(
        "--encode",
        nargs="*",
        choices=["nii", "snirf", "gii", "mat"],
        help="re-encode these modality payloads into binary JData attachments "
        "named <sha256>_<codec>.<bnii|bnirs|bgii|jdb>, instead of referencing "
        "the original file. Requires reading (and rewriting) every payload.",
    )
    conv.add_argument(
        "--encode-codec",
        default="zlib",
        choices=["zlib", "lzma", "lz4", "blosc2zstd", "blosc2lz4", "none"],
        help="compression inside the attachment (default zlib; blosc2zstd is "
        "both smaller and faster)",
    )
    conv.add_argument(
        "--encode-threads",
        type=int,
        default=1,
        help="threads used inside the compressor for one attachment; zlib is "
        "parallelised block-wise, so this scales nearly linearly",
    )
    conv.add_argument("--max-encode", type=int, help="skip payloads larger than this")
    conv.add_argument("--max-doc", type=int, help="document size budget in bytes")
    conv.add_argument("--no-split", action="store_true", help="keep derivatives in the main doc")
    conv.add_argument("--force", action="store_true", help="rewrite even if unchanged")
    conv.add_argument("--log", help="append one JSON result line per dataset here")
    conv.add_argument("--verbose", action="store_true")
    conv.set_defaults(func=cmd_convert)

    push = sub.add_parser("push", help="publish digests through the update handler")
    push.add_argument("--output", required=True)
    push.add_argument("--db", required=True)
    push.add_argument("--split-db", help="database for split-out subtrees")
    push.add_argument("--server", required=True)
    push.add_argument("--design", default="qq")
    push.add_argument("--netrc", default="neurojson.io", help="netrc machine for credentials")
    push.add_argument("--ds", nargs="*")
    push.add_argument("--cas", help="content store, needed to hold trimmed subtrees")
    push.add_argument("--algo", default="md5", choices=["md5", "sha256"])
    push.add_argument("--cas-url", help="base URL template for _DataLink_")
    push.add_argument(
        "--max-trim",
        type=int,
        default=8,
        help="how many times to shed the largest subtree and retry when the "
        "server rejects a document as too large",
    )
    push.add_argument("--allow-production", action="store_true")
    push.add_argument("--verbose", action="store_true")
    push.set_defaults(func=cmd_push)

    dep = sub.add_parser("deploy", help="create databases and install the design document")
    dep.add_argument("--server", required=True)
    dep.add_argument("--db", required=True)
    dep.add_argument("--split-db", help="database for split-out subtrees, e.g. derivatives")
    dep.add_argument("--design", required=True, help="directory of view_*.js files")
    dep.add_argument("--name", default="qq", help="design document name")
    dep.add_argument("--design-name", default="qq", help="design doc used for registry writes")
    dep.add_argument("--netrc", default="neurojson.io")
    dep.add_argument("--admin", nargs="*", help="usernames to set as database admins")
    dep.add_argument("--member", nargs="*", help="usernames to set as database members")
    dep.add_argument("--register", action="store_true", help="add to sys/registry")
    dep.add_argument("--register-name")
    dep.add_argument("--register-url")
    dep.add_argument("--register-datatype", nargs="*")
    dep.add_argument("--warm", action="store_true")
    dep.add_argument("--allow-production", action="store_true")
    dep.set_defaults(func=cmd_deploy)

    views = sub.add_parser("views", help="install a design document from a directory")
    views.add_argument("--db", required=True)
    views.add_argument("--design", required=True, help="directory of view_*.js files")
    views.add_argument("--server", required=True)
    views.add_argument("--name", default="qq")
    views.add_argument("--netrc", default="neurojson.io")
    views.add_argument("--warm", action="store_true", help="build each view after install")
    views.add_argument("--allow-production", action="store_true")
    views.set_defaults(func=cmd_views)

    cas = sub.add_parser("cas", help="inspect or verify the content store")
    cas.add_argument("action", choices=["usage", "verify"])
    cas.add_argument("--cas", required=True)
    cas.add_argument("--sample", type=int, default=200)
    cas.set_defaults(func=cmd_cas)

    ver = sub.add_parser("verify", help="reconcile archive, manifest and content store")
    ver.add_argument("--output", required=True)
    ver.add_argument("--cas")
    ver.add_argument(
        "--algo", default="md5", choices=["md5", "sha256"], help="store hash algorithm"
    )
    ver.add_argument("--ds", nargs="*")
    ver.add_argument("--deep", action="store_true", help="also re-hash stored objects")
    ver.add_argument("--sample", type=int, default=200)
    ver.add_argument("--verbose", action="store_true")
    ver.set_defaults(func=cmd_verify)

    doi = sub.add_parser("doi", help="emit DataCite metadata per dataset version")
    doi.add_argument("--output", required=True)
    doi.add_argument("--db", required=True)
    doi.add_argument("--ds", nargs="*")
    doi.add_argument("--publisher", default="NeuroJSON")
    doi.add_argument("--landing-base", default="https://neurojson.io/db")
    doi.add_argument("--year", type=int)
    doi.add_argument("--stdout", action="store_true", help="print instead of writing files")
    doi.add_argument("--verbose", action="store_true")
    doi.set_defaults(func=cmd_doi)

    rep = sub.add_parser("report", help="summarise converted output")
    rep.add_argument("--output", required=True)
    rep.add_argument("--ds", nargs="*")
    rep.add_argument("--top", type=int, default=25)
    rep.add_argument("--max-doc", type=int, default=7_500_000)
    rep.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
