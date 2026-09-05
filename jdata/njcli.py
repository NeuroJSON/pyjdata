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


def dataset_outdir(outputroot, dsname):
    """Directory holding one dataset's converted output.

    One current document per dataset, not a tree of versions.  CouchDB produces
    a hash for every revision it stores, so it is already the version authority;
    keeping a parallel per-version archive here would mean maintaining a second
    versioning scheme and keeping the two in step.  What the document carries
    instead are the upstream identifiers -- git commit, release tag, per-file
    content hashes -- which is what maps it back to git-annex.
    """
    return os.path.join(outputroot, dsname)


def _write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fid:
        fid.write(text)
    os.replace(tmp, path)


def convert_dataset(dspath, outputroot, dbname, casroot, options, force=False):
    """Convert one dataset and write its output.

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
    label = result["version"].get("VersionLabel") or version or "unknown"
    outdir = dataset_outdir(outputroot, dsname)
    docpath = os.path.join(outdir, "doc.json")
    doctext = canonical_json(result["doc"])

    summary = {
        "ds": dsname,
        "version": version,
        "label": label,
        "commit": result["version"].get("SourceCommit"),
        "docbytes": len(doctext),
        "files": len(result["manifest"]),
        "errors": len(result["errors"]),
        "offloaded": len(result["stats"].get("offloaded", [])),
    }

    if (not force) and os.path.exists(docpath):
        with open(docpath, "r", encoding="utf-8") as fid:
            if fid.read() == doctext:
                summary["status"] = "unchanged"
                summary["seconds"] = time.time() - started
                return summary

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
                "label": label,
                "version": result["version"],
                "stats": result["stats"],
                "errors": result["errors"],
                "docbytes": len(doctext),
                "split": {k: len(canonical_json(v)) for k, v in result["split"].items()},
            },
            indent=1,
            sort_keys=True,
        ),
    )
    summary["status"] = "written"
    summary["seconds"] = time.time() - started
    return summary


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

    if getattr(args, "ai_summary", None):
        options["njbids"]["ai_summary"] = args.ai_summary
    # not conditional on --ai-summary: nesting it there silently pinned every
    # other run to one thread whatever --encode-threads said
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
    summary = {"written": 0, "unchanged": 0, "failed": 0, "conflict": 0}
    results = []
    logfid = open(args.log, "a", encoding="utf-8") if args.log else None

    def record(res):
        results.append(res)
        summary[res.get("status", "failed")] = summary.get(res.get("status"), 0) + 1
        if logfid:
            logfid.write(json.dumps(res) + "\n")
            logfid.flush()
        done = len(results)
        if res.get("status") in ("failed", "conflict"):
            print(
                "  [%d/%d] %-12s %s %s"
                % (done, len(jobs), res["ds"], res["status"].upper(), res.get("error"))
            )
        elif args.verbose or done % 25 == 0 or done == len(jobs):
            print(
                "  [%d/%d] %-12s %-9s %-14s %8.1f kB %5d files %6.1fs%s"
                % (
                    done,
                    len(jobs),
                    res["ds"],
                    res["status"],
                    (res.get("label") or res.get("version") or "-"),
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
        "\n%d written, %d unchanged, %d failed, %d conflicting in %.1fs "
        "(%.1f ds/min, %.1f MB of JSON)"
        % (
            summary.get("written", 0),
            summary.get("unchanged", 0),
            summary.get("failed", 0),
            summary.get("conflict", 0),
            elapsed,
            60.0 * len(jobs) / elapsed if elapsed else 0,
            bytes_out / 1e6,
        )
    )
    bad = [r for r in results if r.get("status") in ("failed", "conflict")]
    for res in bad[:10]:
        print("FAILED %s: %s" % (res["ds"], res.get("error")))
    return 1 if bad else 0


#: files in a version directory that are not documents to publish
NON_DOCUMENT_FILES = ("doc.json", "meta.json", "datacite.json")


def _iter_published(outputroot, names=None, split_names=None):
    """Yield ``(dsname, label, docpath, splitpaths)`` for each converted dataset.

    Split documents are recognised by name against the known split directories,
    not by "any .json that is not doc.json".  The loose test previously picked up
    the DataCite record written alongside the document and tried to publish it
    as a derivatives document.
    """
    known = tuple(split_names or NJBIDS_DEFAULT["split_dirs"])
    for dsname in sorted(names or os.listdir(outputroot)):
        dsdir = os.path.join(outputroot, dsname)
        docpath = os.path.join(dsdir, "doc.json")
        if not os.path.isfile(docpath):
            continue
        label = "-"
        metapath = os.path.join(dsdir, "meta.json")
        if os.path.isfile(metapath):
            try:
                with open(metapath, "r", encoding="utf-8") as fid:
                    label = json.load(fid).get("label") or "-"
            except Exception:
                pass
        splits = {
            name[: -len(".json")]: os.path.join(dsdir, name)
            for name in sorted(os.listdir(dsdir))
            if name.endswith(".json")
            and name not in NON_DOCUMENT_FILES
            and name[: -len(".json")] in known
        }
        yield dsname, label, docpath, splits


def _trim_to(docpath, cas, db, ds, target, limit=None, cas_url_base=None):
    """Shed largest-first until the document is at most ``target`` bytes.

    Returns the list of shed subtrees.  Shedding one subtree per round trip is
    correct but slow to converge: a document of 89 MB against an 8 MB limit
    needs about twenty rejected uploads to get there, each re-sending most of
    the document.  Given a size the server has already refused, shedding down to
    a fraction of it converges in a handful of attempts instead.

    All the shedding for one round happens in memory and the document is written
    once, because re-reading and re-serialising an 89 MB document per subtree is
    itself slower than the upload it is trying to avoid.
    """
    from .njbids import BUDGET_PROTECTED, canonical_json
    from .njcas import cas_url

    with open(docpath, "r", encoding="utf-8") as fid:
        doc = json.load(fid)

    # size every candidate once, then shed in descending order
    candidates = []
    for key, value in doc.items():
        if key in BUDGET_PROTECTED or not isinstance(value, dict):
            continue
        if "_DataLink_" in value:
            continue
        candidates.append((len(canonical_json(value)), key))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    total = len(canonical_json(doc))
    shed = []
    for was, key in candidates:
        if total <= target or (limit is not None and len(shed) >= limit):
            break
        digest, size = cas.put_bytes(canonical_json(doc[key]))
        link = {
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
        doc[key] = link
        total -= was - len(canonical_json(link))
        shed.append({"path": key, "was": was})

    if shed:
        text = canonical_json(doc)
        _write_atomic(docpath, text)
        for item in shed:
            item["now"] = len(text)
    return shed


def _trim_largest_subtree(docpath, cas, db, ds, cas_url_base=None):
    """Shed exactly the largest top-level subtree, or return None if none can be."""
    shed = _trim_to(docpath, cas, db, ds, target=-1, limit=1, cas_url_base=cas_url_base)
    return shed[0] if shed else None


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
    couch = CouchDB(args.server, netrc_machine=args.netrc, timeout=args.timeout, retries=1)
    cas = CAS(args.cas, algo=args.algo) if args.cas else None

    items = list(_iter_published(args.output, args.ds))
    print("pushing %d document(s) to %s/%s" % (len(items), couch.url, args.db))
    ok = failed = trimmed = 0
    # The smallest size this server has actually refused, learned during the run.
    # The first oversized document pays the discovery cost; later ones are
    # trimmed before the attempt rather than after, which matters because an
    # 89 MB upload takes minutes just to be refused.  Still empirical: the bound
    # comes from the server, not from a guess about its internals.
    refused_at = None
    for dsname, version, docpath, splits in items:
        notes = []
        if cas is not None and refused_at is not None and os.path.getsize(docpath) >= refused_at:
            shed = _trim_to(
                docpath,
                cas,
                args.db,
                dsname,
                int(refused_at * args.trim_factor),
                cas_url_base=args.cas_url,
            )
            if shed:
                trimmed += len(shed)
                notes.append(
                    "pre-trimmed %d subtree(s) to %.0fkB"
                    % (len(shed), os.path.getsize(docpath) / 1024.0)
                )
        published = False
        for attempt in range(args.max_trim + 1):
            try:
                couch.push_file(args.db, dsname, docpath, design=args.design, handler=args.handler)
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
                refused = os.path.getsize(docpath)
                refused_at = refused if refused_at is None else min(refused_at, refused)
                shed = _trim_to(
                    docpath,
                    cas,
                    args.db,
                    dsname,
                    int(refused * args.trim_factor),
                    cas_url_base=args.cas_url,
                )
                if not shed:
                    print("  %-12s too large, nothing left to trim" % dsname)
                    break
                trimmed += len(shed)
                notes.append(
                    "%d subtree(s) %.0fkB -> %.0fkB"
                    % (len(shed), refused / 1024.0, os.path.getsize(docpath) / 1024.0)
                )

        if not published:
            failed += 1
            continue

        for name, path in splits.items():
            target = args.split_db or ("%s_%s" % (args.db, name.rstrip("s")))
            try:
                couch.push_file(target, dsname, path, design=args.design, handler=args.handler)
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
    """Check that a converted dataset's links resolve in the content store.

    There is no document-level fingerprint to recompute -- CouchDB produces the
    revision hash, and the document carries upstream identifiers rather than one
    of its own.  What is worth checking is that those identifiers are honoured:

    * every ``_DataLink_`` names an object that is actually present, so the
      links are not promises the store cannot keep;
    * every manifest line agrees with the object it names;
    * with ``--deep``, a sample of objects is re-hashed, proving the bytes are
      intact and (via ``st_nlink``) that the store still holds hardlinks rather
      than private copies.
    """
    cas = CAS(args.cas, mode="none", algo=args.algo, annex_hash=True) if args.cas else None
    checked = missing = unfetched = 0
    problems = []

    for dsname, label, docpath, _splits in _iter_published(args.output, args.ds):
        with open(docpath, "r", encoding="utf-8") as fid:
            doc = json.load(fid)
        checked += 1
        if cas is None:
            continue
        absent = []
        for algo, digest, where in _iter_links(doc):
            if not cas.has(digest):
                # a link to content the mirror has not fetched is expected, not
                # an error: the digest still identifies it for later retrieval
                unfetched += 1
                absent.append(where)
        if absent:
            missing += len(absent)
            problems.append(
                "%s@%s: %d link(s) not present in the store, e.g. %s"
                % (dsname, label, len(absent), absent[0])
            )
        if args.verbose:
            print("  %-12s %-24s %s" % (dsname, label, "ok" if not absent else "see below"))

    print("%d dataset(s) checked, %d link(s) not present in the store" % (checked, missing))
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
    for dsname, label, docpath, _splits in _iter_published(args.output, args.ds):
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
                print("  %-12s %-22s %s" % (dsname, label, record.get("titles")[0]["title"][:50]))
    if not args.stdout:
        print("wrote %d datacite.json record(s)" % written)
    return 0


def cmd_report(args):
    rows = []
    for dsname, label, docpath, splits in _iter_published(args.output, args.ds):
        metapath = os.path.join(os.path.dirname(docpath), "meta.json")
        meta = {}
        if os.path.isfile(metapath):
            with open(metapath, "r", encoding="utf-8") as fid:
                meta = json.load(fid)
        rows.append(
            {
                "ds": dsname,
                "label": label,
                "docbytes": os.path.getsize(docpath),
                "files": meta.get("stats", {}).get("files"),
                "errors": len(meta.get("errors", [])),
                "offloaded": len(meta.get("stats", {}).get("offloaded", [])),
                "commit": (meta.get("version", {}) or {}).get("SourceCommit") or "",
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
        "%-12s %-22s %10s %7s %7s %6s  %s"
        % ("dataset", "label", "docbytes", "files", "errors", "offl", "commit")
    )
    for row in rows[: args.top]:
        print(
            "%-12s %-22s %10d %7s %7d %6d  %s"
            % (
                row["ds"],
                row["label"],
                row["docbytes"],
                row["files"],
                row["errors"],
                row["offloaded"],
                row["commit"][:12],
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
        choices=["nii", "snirf", "gii", "mat", "eeg"],
        help="re-encode these modality payloads into binary JData attachments "
        "named <sha256>_<codec>.<bnii|bnirs|bgii|jdb|jeeg>, instead of referencing "
        "the original file. Requires reading (and rewriting) every payload.",
    )
    conv.add_argument(
        "--ai-summary",
        metavar="DIR",
        help="directory of <dataset>.ai.json files whose .datainfo contents are "
             "merged into the document's .neurojson metadata (AISummary and "
             "Citation.cff). Missing or unreadable files are skipped silently.",
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
    push.add_argument(
        "--split-db",
        help="database receiving split-out subtrees such as derivatives. "
        "Defaults to <db>_derivative, which is only right if the databases were "
        "named that way -- pass it explicitly otherwise",
    )
    push.add_argument("--server", required=True)
    push.add_argument("--design", default="qq")
    push.add_argument(
        "--handler",
        default="replace",
        help="update handler to publish through. 'replace' (default) replaces "
        "the document body, which is what a digest needs; 'timestamp' merges, "
        "which would keep data that was removed upstream",
    )
    push.add_argument("--netrc", default="neurojson.io", help="netrc machine for credentials")
    push.add_argument("--ds", nargs="*")
    push.add_argument("--cas", help="content store, needed to hold trimmed subtrees")
    push.add_argument("--algo", default="md5", choices=["md5", "sha256"])
    push.add_argument("--cas-url", help="base URL template for _DataLink_")
    push.add_argument(
        "--max-trim",
        type=int,
        default=12,
        help="how many rounds of trim-and-retry to allow when the server "
        "rejects a document as too large",
    )
    push.add_argument(
        "--trim-factor",
        type=float,
        default=0.5,
        help="each round sheds largest-first until the document is at most this "
        "fraction of the size just refused (default 0.5, so it converges in a "
        "few rounds rather than one subtree per round trip)",
    )
    push.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-request timeout; a document the server will refuse can "
        "otherwise sit in a blocked upload for a long time",
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
