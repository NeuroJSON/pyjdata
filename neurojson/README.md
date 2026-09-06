# NeuroJSON deployment assets

Files here are **not** part of the importable `jdata` package.  They are the
server-side assets and helper scripts for running the NeuroJSON publication
pipeline, kept in this repository so that changes to them are reviewable and
diffable.

```
SCHEMA.md                conversion rules, output schema, view/handler changes
design/qq/               CouchDB design document
design/frozen/           the design document as deployed for the legacy
                         `openneuro` database, kept verbatim for diffing
simulate.js              runs design-document JavaScript outside CouchDB
pgcompat.js              runs the Postgres search layer's view ports against a
                         document, to catch schema drift between the two
njsync.sh                mirror update and machine-readable change detection
njpipeline.sh            the daily six-stage run, for cron
njpipeline.conf.example  every setting, documented
```

## Why the design document lives here

A CouchDB design document is JavaScript executed by the database server.  Before
this directory existed, the live views existed *only* on the server -- there was
no copy in any repository, so a view change could not be reviewed, diffed or
rolled back, and the Postgres search layer maintains hand-written ports of the
same map functions that "drift silently" (its own words) when the originals
change.  Keeping the source here makes both sides diffable against one another.

## design/qq

| file | role |
| --- | --- |
| `view_dbinfo.js` | one row per dataset: name, description, subjects, modalities, and the version/fingerprint block |
| `view_subjects.js` | one row per subject, keyed for range queries on age, sex, session/modality/task/run counts |
| `view_participantsfields.js` | the set of `participants.tsv` column names in use |
| `view_links.js` | one row per distinct `_DataLink_`, keyed `[doc._id, ext, size]`, with the content hash |
| `view_versions.js` | one row per published dataset version, for reconciling the on-disk archive against the database |
| `view_updatetime.js` | publication times, for "recently updated" listings |
| `update_timestamp.js` | the update handler every write goes through |
| `validate_doc_update.js` | write authorisation |

Differences from `design/frozen`: the metadata key moved from `.datainfo` to
`.neurojson`; `update_timestamp.js` sets `CreateTime` from the server clock
rather than a hardcoded constant and refuses to let a client overwrite it;
`view_links.js` also emits the content hash; and `view_versions.js` is new.

`view_links.js` keeps the `[doc._id, ext, size]` key shape, because the Postgres
sync range-queries one dataset at a time with it -- fetching the whole links
view of a large database exceeds Node's maximum string size.

## Installing

```bash
python3 -m jdata.njcli views --server http://host:5984 --db openneuro_full \
        --design neurojson/design/qq --warm
```

## Testing without a server

```bash
node neurojson/simulate.js view   neurojson/design/qq dbinfo    path/to/doc.json
node neurojson/simulate.js update neurojson/design/qq timestamp path/to/doc.json
```

`test/testnjviews.py` drives both of these.

---

# Running the pipeline

## One-time setup

```bash
pip install -e ~/space/git/Project/github/pyjdata     # or set PYTHONPATH
cp njpipeline.conf.example njpipeline.conf            # then edit
chmod 600 ~/.netrc                                    # python's netrc requires this
```

Add a credential for the target server to `~/.netrc`:

```
machine zodiac login admin password <password>
```

Create the databases, install the views and register them for search.  This
needs a CouchDB **server admin** credential; a database admin cannot create
databases.

```bash
python3 -m jdata.njcli deploy \
    --server http://zodiac:5555 \
    --db openneuro_full --split-db openneuro_derivative \
    --design neurojson/design/qq \
    --netrc zodiac --admin fangq --register --warm
```

Registration is not optional if you want search to work: the Postgres sync
discovers which databases to index by reading `sys/registry`.

## Daily run

```bash
./njpipeline.sh              # sync, convert, publish, refresh views, sync search
./njpipeline.sh --dry-run    # plan only
./njpipeline.sh --no-sync    # convert and publish without touching the mirror
```

As a cron entry:

```
15 3 * * * /home/users/fangq/space/git/Project/github/pyjdata/neurojson/njpipeline.sh
```

## Stages individually

```bash
# 1-2  refresh the mirror and write log/<date>_changed.json
./njsync.sh all
./njsync.sh status

# 3    convert (per-dataset process parallelism)
python3 -m jdata.njcli convert \
    --input  /lake/neurojson/prep/openneuro_full/orig \
    --output /lake/neurojson/prep/openneuro_full/json \
    --db openneuro_full --cas /lake/neurojson/cas \
    --threads 24 --log convert.jsonl

# 4    publish (POST through the update handler; never a PUT)
python3 -m jdata.njcli push --output ... --db openneuro_full \
    --split-db openneuro_derivative --server http://zodiac:5555 --netrc zodiac

# 5    refresh and warm the views
python3 -m jdata.njcli views --db openneuro_full \
    --design neurojson/design/qq --server http://zodiac:5555 --warm

# 6    Postgres search sync (separate Node service, follows _changes)
cd <NeuroJSON_io>/backend && node sync/incrementalSync.js
```

## Inspecting and checking

```bash
python3 -m jdata.njcli report --output ...          # sizes, errors, budget offloads
python3 -m jdata.njcli verify --output ... --cas ... --deep
python3 -m jdata.njcli cas usage  --cas ...        # objects and space actually used
python3 -m jdata.njcli doi --output ... --db openneuro_full
```

`verify` is the one to run before minting DOIs.  It recomputes each version's
fingerprint from the archive, confirms every `_DataLink_` resolves in the store,
and with `--deep` re-hashes a sample of objects.

## Conversion is safe to re-run

Conversion is idempotent and content-addressed:

* a dataset whose document is byte-identical to the one on disk is reported
  `unchanged` and nothing is written;
* the hash memo is keyed by the git-annex key, itself a content hash, so a file
  that is unchanged (including across dataset versions) is never re-hashed;
* store objects are created with `os.link` and skipped if already present.

A warm re-run of a converted dataset takes milliseconds.  The first pass over a
large mirror is dominated by hashing every payload once.

## Output layout

```
<output>/<ds>/<version>/doc.json        immutable, byte-reproducible
<output>/<ds>/<version>/manifest.tsv    <sha256> <size> <relpath>, sorted by path
<output>/<ds>/<version>/derivatives.json
<output>/<ds>/<version>/meta.json       version, fingerprint, stats, errors
<output>/<ds>/<version>/datacite.json   written by `njcli doi`
<output>/<ds>/latest -> <version>
```

The per-version directory is the DOI target and the source of truth.  CouchDB
holds the latest version only: compaction discards old revision bodies, so a
`_rev` cannot anchor a DOI on its own.

## Content store

```
<cas>/objects/<h0:2>/<h2:4>/<sha256>    the payload, hardlinked
<cas>/index.sqlite                      sha256 memo, keyed by git-annex key
```

The store must be on the same filesystem as the mirror, otherwise objects become
copies instead of hardlinks and the space cost goes from nothing to a second
full copy.  `njcli cas usage` reports `exclusive_bytes`, which should be ~0.

Serving it needs no CGI; mod_rewrite maps the query parameters to a path.

A link names **two** things: the source content hash, and -- when the payload
was re-encoded -- the derived form stored beside it as `<hash><enc>`, e.g.
`<hash>_zlib.bnii`.  A rule that reads only `hash=` serves the bare object,
which for an encoded payload frequently does not exist: measured over 600 real
links from 25 datasets, hash-only resolved **69.7%** and left the rest dead,
while the `enc`-aware rule below resolved **600 of 600**.

```apache
Alias /neurojson-cas /path/to/cas/objects

# 1. digest and its two shard bytes
RewriteCond %{QUERY_STRING} (^|&)hash=sha256:((..)(..)[0-9a-f]{60})(&|$)
RewriteRule ^/io/cas\.cgi$ - [E=CAS_H:%2,E=CAS_A:%3,E=CAS_B:%4]

# 2. the derived encoding, when the document names one.  The pattern is a
#    whitelist, so a crafted enc= cannot walk out of the store; anything it
#    rejects falls through to the bare object rather than to a path outside it.
RewriteCond %{QUERY_STRING} (^|&)enc=(_[A-Za-z0-9]+\.[A-Za-z0-9]+)(&|$)
RewriteRule ^/io/cas\.cgi$ - [E=CAS_E:%2]

# 3. serve.  An unset CAS_E expands empty, which is exactly the bare object.
RewriteCond %{ENV:CAS_H} ^[0-9a-f]{64}$
RewriteRule ^/io/cas\.cgi$ /neurojson-cas/%{ENV:CAS_A}/%{ENV:CAS_B}/%{ENV:CAS_H}%{ENV:CAS_E} [L]

<Directory /path/to/cas/objects>
  Require all granted
  Options -Indexes -FollowSymLinks +SymLinksIfOwnerMatch
  Header always set Cache-Control "public, max-age=31536000, immutable"
  Header always set Access-Control-Allow-Origin "*"
</Directory>
```

Two captures cannot share one `RewriteCond` backreference -- `%N` refers to the
last condition that matched -- hence the env variables.

Because the identifier is a content hash the response is immutable, so it can be
cached forever and served with byte ranges.  Ranges matter more than they look:
an encoded array carries `_ArrayZipOffsets_`, so a client that wants one block
of a 500 MB attachment can ask for exactly that block.

### What a link yields

For an encoded payload the response is the **JData attachment**, not the
original file, even though `file=` names the original.  That is the intended
contract: the metadata sitting beside `_DataLink_` in the document is
transitional -- enough to search, annotate and comment on -- and a resolver
replaces the whole node with the fetched object.

Whether the original is *also* retrievable depends on the handler that stored
it: 222 of 399 sampled encoded links had their source object present as well,
so `enc=` should be treated as the authoritative form and the bare hash as a
bonus, not the other way round.

## Safety

`deploy` and `push` refuse to write to `neurojson.io` or `neurojson.org` unless
given `--allow-production`.  The production admin credential is commonly present
in the environment, and publishing is not trivially reversible.

`CouchDB.__init__` strips any credentials out of a URL it is given, so they
cannot reappear in a log line or an exception message.
