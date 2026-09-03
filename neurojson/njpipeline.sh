#!/bin/bash
#
# njpipeline.sh -- the daily NeuroJSON publication run.
#
# Chains the six stages: sync the mirror, work out what changed, convert the
# changed datasets, publish them, refresh the views, and hand off to the
# Postgres search sync.
#
#   njpipeline.sh            full run
#   njpipeline.sh --dry-run  plan only; no downloads, no writes
#   njpipeline.sh --no-sync  skip the mirror update (convert/publish only)
#
# Configuration comes from njpipeline.conf next to this script, if present,
# otherwise from the environment:
#
#   NJ_ROOT        mirror + output root  (default /lake/neurojson/prep/openneuro_full)
#   NJ_CAS         content store root    (default /lake/neurojson/cas)
#   NJ_DB          CouchDB database      (default openneuro_full)
#   NJ_SPLIT_DB    database for derivatives (default ${NJ_DB}_derivative)
#   NJ_SERVER      CouchDB base URL      (no default; must be set to publish)
#   NJ_NETRC       netrc machine for credentials (default the NJ_SERVER host)
#   NJ_THREADS     conversion workers    (default 24)
#   NJ_PYTHONPATH  where jdata lives, if not installed
#
# Why the stages are chained rather than merged: conversion is expensive and
# happens where the data is, publication is a network operation that may fail
# or be refused, and the Postgres sync is a separate service.  A failure in a
# later stage must never invalidate the work of an earlier one, and the
# immutable per-version output on disk is the source of truth throughout.

set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/njpipeline.conf" ] && . "$HERE/njpipeline.conf"

NJ_ROOT="${NJ_ROOT:-/lake/neurojson/prep/openneuro_full}"
NJ_CAS="${NJ_CAS:-/lake/neurojson/cas}"
NJ_DB="${NJ_DB:-openneuro_full}"
NJ_SPLIT_DB="${NJ_SPLIT_DB:-${NJ_DB}_derivative}"
NJ_SERVER="${NJ_SERVER:-}"
NJ_THREADS="${NJ_THREADS:-24}"
NJ_DESIGN="${NJ_DESIGN:-$HERE/design/qq}"
LOG="$NJ_ROOT/log"
DATE="$(date +%Y%m%d)"
RUNLOG="$LOG/pipeline_${DATE}.log"
LOCK="$NJ_ROOT/.njpipeline.lock"

DRY=0
DO_SYNC=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        --no-sync) DO_SYNC=0 ;;
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

[ -n "${NJ_PYTHONPATH:-}" ] && export PYTHONPATH="$NJ_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
NJCLI=(python3 -m jdata.njcli)

mkdir -p "$LOG"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUNLOG"; }
die() { say "FATAL: $*"; exit 1; }

exec 9>"$LOCK"
if ! flock -n 9; then
    say "another njpipeline run holds $LOCK; exiting"
    exit 0
fi

say "=== njpipeline start (root=$NJ_ROOT db=$NJ_DB dry=$DRY) ==="

# ---- stage 1+2: mirror update and change detection -----------------------

CHANGED="$LOG/${DATE}_changed.json"
if [ "$DO_SYNC" = "1" ]; then
    say "stage 1/2: sync mirror and detect changes"
    if [ "$DRY" = "1" ]; then
        NJ_DRY_RUN=1 "$HERE/njsync.sh" fetch >>"$RUNLOG" 2>&1 || say "WARN: fetch reported errors"
        NJ_DRY_RUN=1 "$HERE/njsync.sh" diff  >>"$RUNLOG" 2>&1 || die "diff failed"
    else
        "$HERE/njsync.sh" all >>"$RUNLOG" 2>&1 || say "WARN: njsync reported errors"
    fi
else
    say "stage 1/2: skipped (--no-sync)"
fi

# ---- stage 3: convert -----------------------------------------------------

DSLIST=()
if [ -f "$CHANGED" ]; then
    while IFS= read -r ds; do [ -n "$ds" ] && DSLIST+=("$ds"); done < <(
        python3 -c '
import json, sys
for item in json.load(open(sys.argv[1]))["changed"]:
    print(item["ds"])
' "$CHANGED"
    )
fi

if [ "${#DSLIST[@]}" -eq 0 ]; then
    say "stage 3: no changed datasets; nothing to convert"
else
    say "stage 3: converting ${#DSLIST[@]} dataset(s) with $NJ_THREADS workers"
    if [ "$DRY" = "1" ]; then
        say "  dry run: ${DSLIST[*]}"
    else
        "${NJCLI[@]}" convert \
            --input "$NJ_ROOT/orig" --output "$NJ_ROOT/json" \
            --db "$NJ_DB" --cas "$NJ_CAS" \
            --ds "${DSLIST[@]}" --threads "$NJ_THREADS" \
            --log "$LOG/convert_${DATE}.jsonl" >>"$RUNLOG" 2>&1 \
            || say "WARN: some datasets failed to convert; see $LOG/convert_${DATE}.jsonl"
    fi
fi

# ---- stage 4+5: publish and refresh views --------------------------------

if [ -z "$NJ_SERVER" ]; then
    say "stage 4/5: skipped (NJ_SERVER not set)"
elif [ "$DRY" = "1" ]; then
    say "stage 4/5: dry run; would publish to $NJ_SERVER/$NJ_DB"
elif [ "${#DSLIST[@]}" -eq 0 ]; then
    say "stage 4/5: nothing new to publish"
else
    NETRC="${NJ_NETRC:-$(python3 -c '
import sys, urllib.parse
print(urllib.parse.urlsplit(sys.argv[1]).hostname or "")
' "$NJ_SERVER")}"
    say "stage 4: publishing ${#DSLIST[@]} document(s) to $NJ_SERVER/$NJ_DB"
    "${NJCLI[@]}" push \
        --output "$NJ_ROOT/json" --db "$NJ_DB" --split-db "$NJ_SPLIT_DB" \
        --server "$NJ_SERVER" --netrc "$NETRC" \
        --ds "${DSLIST[@]}" >>"$RUNLOG" 2>&1 \
        || say "WARN: some documents failed to publish"

    say "stage 5: refreshing views"
    "${NJCLI[@]}" views \
        --db "$NJ_DB" --design "$NJ_DESIGN" --server "$NJ_SERVER" \
        --netrc "$NETRC" --warm >>"$RUNLOG" 2>&1 \
        || say "WARN: view refresh reported errors"
fi

# ---- stage 6: Postgres search sync ---------------------------------------

# The search layer is a separate Node service (backend/sync/incrementalSync.js)
# that follows CouchDB's _changes feed, so it only needs to be nudged.  It
# discovers databases from sys/registry, which `njcli deploy --register` sets up.
if [ -n "${NJ_PGSYNC_DIR:-}" ] && [ "$DRY" != "1" ]; then
    say "stage 6: postgres search sync"
    ( cd "$NJ_PGSYNC_DIR" && node sync/incrementalSync.js ) >>"$RUNLOG" 2>&1 \
        || say "WARN: postgres sync reported errors"
else
    say "stage 6: skipped (set NJ_PGSYNC_DIR to the backend checkout to enable)"
fi

# ---- summary --------------------------------------------------------------

say "stage 7: summary"
"${NJCLI[@]}" report --output "$NJ_ROOT/json" --top 10 >>"$RUNLOG" 2>&1 || true
if [ "$DRY" != "1" ]; then
    "${NJCLI[@]}" cas usage --cas "$NJ_CAS" >>"$RUNLOG" 2>&1 || true
fi
say "=== njpipeline done; log $RUNLOG ==="
