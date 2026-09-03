# NeuroJSON deployment assets

Files here are **not** part of the importable `jdata` package.  They are the
server-side assets and helper scripts for running the NeuroJSON publication
pipeline, kept in this repository so that changes to them are reviewable and
diffable.

```
design/qq/        CouchDB design document for the version-invariant schema
design/frozen/    the design document as deployed for the legacy `openneuro`
                  database, kept verbatim for reference and diffing
simulate.js       runs design-document JavaScript outside CouchDB
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
