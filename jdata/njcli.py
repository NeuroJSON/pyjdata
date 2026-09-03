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
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from .njcas import CAS
from .njbids import bids2json, canonical_json

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
    cas = CAS(casroot, mode=options.get("cas_mode", "link"))
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

    options = {"cas_mode": args.cas_mode, "njbids": {}}
    if args.max_doc:
        options["njbids"]["max_doc"] = args.max_doc
    if args.cas_url:
        options["njbids"]["cas_url"] = args.cas_url
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


def _iter_published(outputroot, names=None):
    """Yield ``(dsname, version, docpath, splitpaths)`` for each latest version."""
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
        splits = {
            name[: -len(".json")]: os.path.join(vdir, name)
            for name in sorted(os.listdir(vdir))
            if name.endswith(".json") and name not in ("doc.json", "meta.json")
        }
        yield dsname, version, docpath, splits


def cmd_push(args):
    from .njcouch import CouchDB, CouchError

    couch = CouchDB(args.server, netrc_machine=args.netrc)
    if args.server and "neurojson.io" in args.server and not args.allow_production:
        print(
            "refusing to target neurojson.io without --allow-production",
            file=sys.stderr,
        )
        return 1

    items = list(_iter_published(args.output, args.ds))
    print("pushing %d document(s) to %s/%s" % (len(items), couch.url, args.db))
    ok = failed = 0
    for dsname, version, docpath, splits in items:
        try:
            couch.push_file(args.db, dsname, docpath, design=args.design)
            for name, path in splits.items():
                target = args.split_db or ("%s_%s" % (args.db, name.rstrip("s")))
                couch.push_file(target, dsname, path, design=args.design)
            ok += 1
            if args.verbose:
                print(
                    "  %-12s %-14s %8.1f kB" % (dsname, version, os.path.getsize(docpath) / 1024.0)
                )
        except CouchError as err:
            failed += 1
            print("  %-12s FAILED %s %s" % (dsname, err.status, err.body))
    print("%d pushed, %d failed" % (ok, failed))
    return 1 if failed else 0


def cmd_views(args):
    from .njcouch import CouchDB, design_from_dir

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
    conv.add_argument("--threads", type=int, default=8, help="parallel datasets")
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
    push.add_argument("--allow-production", action="store_true")
    push.add_argument("--verbose", action="store_true")
    push.set_defaults(func=cmd_push)

    views = sub.add_parser("views", help="install a design document from a directory")
    views.add_argument("--db", required=True)
    views.add_argument("--design", required=True, help="directory of view_*.js files")
    views.add_argument("--server", required=True)
    views.add_argument("--name", default="qq")
    views.add_argument("--netrc", default="neurojson.io")
    views.add_argument("--warm", action="store_true", help="build each view after install")
    views.set_defaults(func=cmd_views)

    cas = sub.add_parser("cas", help="inspect or verify the content store")
    cas.add_argument("action", choices=["usage", "verify"])
    cas.add_argument("--cas", required=True)
    cas.add_argument("--sample", type=int, default=200)
    cas.set_defaults(func=cmd_cas)

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
