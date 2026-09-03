#!/bin/bash
#
# njsync.sh -- keep a datalad mirror of a dataset collection up to date and
# report, in machine-readable form, exactly which datasets changed.
#
#   njsync.sh fetch      refresh remote refs and tags for every subdataset
#   njsync.sh diff       write <date>_changed.json listing changed datasets
#   njsync.sh get        fetch annex content for the changed datasets
#   njsync.sh all        fetch, then diff, then get
#   njsync.sh status     summarise the most recent run
#
# Environment:
#   NJ_ROOT      mirror root (default /lake/neurojson/prep/openneuro_full)
#   NJ_FETCH_J   parallel git fetch jobs      (default 12)
#   NJ_GET_J     parallel datalad get jobs    (default 6)
#   NJ_GET_JJ    -J passed to each datalad get (default 3)
#   NJ_DRY_RUN   set to 1 to plan without downloading
#
# Design notes
# ------------
# * A single flock guards the whole run.  A content fetch can outlive the day's
#   interval, and two overlapping runs would race on the same git repositories.
# * The diff stage emits JSON, not shell command lines.  The previous scripts
#   printed `datalad get ...` strings, which meant the converter had no way to
#   learn *which* datasets changed or what they changed from.
# * Concurrency is capped deliberately.  `parallel -j 12` combined with
#   `datalad get -J 5` is up to 60 simultaneous git processes; that is what
#   produced the long sequence of manual `kill -9` calls in the operator's shell
#   history.  The defaults here keep the product at 18.
# * Datasets whose content is not publicly retrievable (a private S3 bucket, a
#   dead export remote) fail identically on every run.  They are recorded once
#   in known_failures.txt and skipped until that file is edited, instead of
#   consuming a slot every night.

set -u -o pipefail

NJ_ROOT="${NJ_ROOT:-/lake/neurojson/prep/openneuro_full}"
ORIG="$NJ_ROOT/orig"
LOG="$NJ_ROOT/log"
LOCK="$NJ_ROOT/.njsync.lock"
FETCH_J="${NJ_FETCH_J:-12}"
GET_J="${NJ_GET_J:-6}"
GET_JJ="${NJ_GET_JJ:-3}"
DATE="$(date +%Y%m%d)"
CHANGED="$LOG/${DATE}_changed.json"
FAILURES="$LOG/known_failures.txt"

export GIT_TERMINAL_PROMPT=0
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=60
export GIT_HTTP_TIMEOUT=300

mkdir -p "$LOG"
touch "$FAILURES"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

require() {
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || { say "ERROR: $tool not found in PATH"; exit 1; }
    done
}

# ---------------------------------------------------------------- fetch

fetch_one() {
    # Refresh one subdataset's remote refs.  --tags matters: dataset versions
    # are published as tags, and without them no semantic version label can be
    # resolved for the converted document.
    local ds="$1"
    [ -e "$ORIG/$ds/.git" ] || { echo "SKIP $ds not-installed"; return 0; }
    if git -C "$ORIG/$ds" fetch --quiet --tags --prune origin 2>/dev/null; then
        echo "OK $ds"
    else
        echo "ERR $ds fetch-failed"
    fi
}
export -f fetch_one
export ORIG

cmd_fetch() {
    require git parallel
    say "fetching refs for subdatasets under $ORIG (-j $FETCH_J)"
    ls -1 "$ORIG" | grep -E '^ds[0-9]+$' \
        | parallel -j "$FETCH_J" --timeout 600 fetch_one {} \
        > "$LOG/${DATE}_fetch.txt" 2>&1
    local ok err skip
    ok=$(grep -c '^OK '   "$LOG/${DATE}_fetch.txt" || true)
    err=$(grep -c '^ERR '  "$LOG/${DATE}_fetch.txt" || true)
    skip=$(grep -c '^SKIP ' "$LOG/${DATE}_fetch.txt" || true)
    say "fetch complete: $ok ok, $err failed, $skip not installed"
}

# ---------------------------------------------------------------- diff

cmd_diff() {
    require git python3
    say "detecting changed datasets"
    # Resolving the remote default branch has to be tolerant: many OpenNeuro
    # mirrors have no origin/HEAD, so fall back through main and master before
    # giving up.
    python3 - "$ORIG" "$CHANGED" "$FAILURES" <<'PYEOF'
import json, os, subprocess, sys

orig, outpath, failpath = sys.argv[1], sys.argv[2], sys.argv[3]
known = set()
if os.path.exists(failpath):
    with open(failpath) as fid:
        known = {l.split("#")[0].strip() for l in fid if l.split("#")[0].strip()}


def git(ds, *args):
    res = subprocess.run(
        ["git", "-C", os.path.join(orig, ds)] + list(args),
        capture_output=True, text=True, timeout=120,
    )
    return res.stdout.strip() if res.returncode == 0 else ""


changed, skipped, unresolved = [], [], []
for ds in sorted(d for d in os.listdir(orig) if d.startswith("ds")):
    if not os.path.exists(os.path.join(orig, ds, ".git")):
        continue
    if ds in known:
        skipped.append(ds)
        continue
    local = git(ds, "rev-parse", "HEAD")
    remote = ""
    for ref in ("refs/remotes/origin/HEAD", "refs/remotes/origin/main",
                "refs/remotes/origin/master"):
        remote = git(ds, "rev-parse", ref)
        if remote:
            break
    if not remote:
        unresolved.append(ds)
        continue
    if local != remote:
        changed.append({
            "ds": ds,
            "old_sha": local,
            "new_sha": remote,
            "kind": "updated" if local else "new",
            "tags": [t for t in git(ds, "tag").splitlines() if t],
        })

report = {
    "generated": None,          # deliberately absent: keeps the file diffable
    "changed": changed,
    "skipped_known_failures": skipped,
    "unresolved_default_branch": unresolved,
}
with open(outpath, "w") as fid:
    json.dump(report, fid, indent=1, sort_keys=True)
print("changed=%d skipped=%d unresolved=%d" % (len(changed), len(skipped), len(unresolved)))
PYEOF
    say "wrote $CHANGED"
}

# ---------------------------------------------------------------- get

cmd_get() {
    require datalad parallel python3
    [ -f "$CHANGED" ] || { say "ERROR: $CHANGED missing; run diff first"; exit 1; }
    local list="$LOG/${DATE}_get.txt"
    python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
for item in report["changed"]:
    print(item["ds"])
' "$CHANGED" > "$list"
    local n
    n=$(wc -l < "$list")
    say "$n dataset(s) to update"
    [ "$n" -eq 0 ] && return 0
    if [ "${NJ_DRY_RUN:-0}" = "1" ]; then
        say "dry run: would merge and get $n dataset(s)"
        cat "$list"
        return 0
    fi
    # merge the fetched refs, then pull content; -j x -J keeps total git
    # processes bounded
    ( cd "$ORIG" && parallel -j "$GET_J" --timeout 7200 \
        "datalad update --how merge -d . {} >/dev/null 2>&1; datalad get -r -J $GET_JJ {}" \
        :::: "$list" ) > "$LOG/${DATE}_downloaded.txt" 2> "$LOG/${DATE}_error.txt"
    say "get complete; see $LOG/${DATE}_downloaded.txt"
    # anything that failed for a non-transient reason is a candidate for the
    # skip list, but promotion is left to a human: a transient network failure
    # must not silently retire a dataset
    if grep -qi 'does not allow public access\|unknown export location' "$LOG/${DATE}_error.txt" 2>/dev/null; then
        say "NOTE: permanent-looking failures present; review $LOG/${DATE}_error.txt"
        say "      add dataset ids to $FAILURES to stop retrying them"
    fi
}

cmd_status() {
    local latest
    latest=$(ls -1t "$LOG"/*_changed.json 2>/dev/null | head -1)
    [ -n "$latest" ] || { say "no run recorded yet"; return 0; }
    say "latest change report: $latest"
    python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
print("  changed:   %d" % len(r["changed"]))
print("  skipped:   %d" % len(r["skipped_known_failures"]))
print("  unresolved:%d" % len(r["unresolved_default_branch"]))
for item in r["changed"][:15]:
    print("    %-12s %s -> %s" % (item["ds"], item["old_sha"][:8], item["new_sha"][:8]))
' "$latest"
}

# ---------------------------------------------------------------- main

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[ $# -ge 1 ] || usage
action="$1"

case "$action" in
    fetch|diff|get|all)
        exec 9>"$LOCK"
        if ! flock -n 9; then
            say "another njsync run holds $LOCK; exiting"
            exit 0
        fi
        ;;
esac

case "$action" in
    fetch)  cmd_fetch ;;
    diff)   cmd_diff ;;
    get)    cmd_get ;;
    all)    cmd_fetch && cmd_diff && cmd_get ;;
    status) cmd_status ;;
    *)      usage ;;
esac
