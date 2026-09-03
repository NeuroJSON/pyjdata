# Conversion rules and output schema

Reference for `jdata.njbids` (conversion), `jdata.njcas` (content store) and
`neurojson/design/qq` (the CouchDB design document).

## 1. What the conversion decides

A dataset is walked depth-first with files before subdirectories, both sorted by
name at every level. That ordering is fixed so the document is byte-reproducible.

Each file becomes one JSON value, placed at the nested location its relative path
describes: `sub-01/anat/x.nii.gz` lands at `doc["sub-01"]["anat"]["x.nii.gz"]`.
Keys are the original filenames, unchanged — the `subjects` view splits them on
`_` to find `task-` and `run-` entities, so they have to stay intact.

### 1.1 Top-level subtrees

| directory | treatment |
| --- | --- |
| `derivatives/` | converted into a **separate document**, published to its own database; the main document keeps a cross-reference |
| `sourcedata/`, `code/`, `stimuli/` | walked but **link-only** — every file becomes a `_DataLink_`, never inlined |
| anything else | converted normally |

Configurable via `split_dirs` and `linkonly_dirs`. The split exists because these
subtrees dominate: `ds003097` is 380k of its 394k files in `derivatives/`, and
`ds005811`'s `stimuli/ImageNet` alone is 19.5 MB of link entries.

### 1.2 Per-file dispatch

Checked in this order; the first match wins. Compound extensions
(`.nii.gz`, `.tsv.gz`, `.csv.gz`, `.img.gz`, `.gii.gz`, `.mgh.gz`) are recognised
whole, so a parent directory containing a dot cannot confuse the match.

| test | handler | inlined | linked |
| --- | --- | --- | --- |
| dangling symlink | — | nothing | annex-key hash + exact size |
| not `stat()`-able | — | nothing | `unreadable:<path>` |
| zero length | — | `{}` | — |
| in a link-only subtree | — | nothing | content hash |
| `.nii .nii.gz .hdr .img .img.gz` | `_nifti` | JNIfTI header (41 fields) | `NIFTIData` |
| `.gii` | `_gifti` | JGIfTI structure | `GIFTIObject` when large or encoded |
| `.snirf` | `_snirf` | metadata, probe, stim, measurement list | `SNIRFObject` |
| `.tsv .csv .tsv.gz .csv.gz` | `_tabular` | column-oriented arrays | whole table if over `max_tsv` |
| `.json .jmsh .jnii .jnirs .jgii .jbids` | `_jsonfile` | parsed JSON | whole file if over `max_json` |
| `.bval .bvec` | `_bvec` | numeric array | whole file if over `max_bvec` |
| `.vhdr .vmrk` | `_brainvision` | parsed INI header, channels, markers | — |
| `.edf .bdf` | `_edf` | EDF/EDF+/BDF header | `EDFObject` |
| `.set` | `_eeglab` | HDF5 or MAT structure summary | `EEGLABObject` |
| `.nwb .h5 .hdf5` | `_hdf5` | structure + attributes | `HDF5Object` |
| `.mat` | `_mat` | MAT variable list, or FSL/VEST header | `MATObject` |
| `README CHANGES LICENSE CITATION AUTHORS TASK`, `.md .txt .rst .m .cff .bib .tex` | `_text` | the text | whole file if over `max_text` |
| anything else | `_safe_link` | nothing | content hash |

`participants.*` and `*_scans.tsv` are **always** inlined regardless of size,
because they drive subject search.

Anything unparseable degrades to a link and is recorded in `errors`. No single
file can abort a dataset — that contract is pinned down by `test/testnjrobust.py`
across broken and looping symlinks, unreadable files, a subject that is a file,
40 levels of nesting, non-ASCII and 200-character filenames, and six payloads
whose contents do not match their extension.

### 1.3 Inline size ceilings

| option | default |
| --- | --- |
| `max_tsv`, `max_json`, `max_text`, `max_bvec` | 1 MiB |
| `max_mat` | 2 MiB |
| `max_h5_elem`, `max_snirf_elem` | 256 elements |
| `max_doc` | 64 MiB |
| `max_leaf_offloads` | 256 |

`max_h5_elem` is an element count, not a byte size, and there is a name-based
denylist as well: SNIRF `dataTimeSeries`, `time` and `dataOffset` are dropped
whatever their size, because a short recording's signal can slip under any
element threshold while still being sample data rather than metadata.

`max_doc` is **not** the CouchDB limit — see §4. It is a sanity bound: one
dataset built a 2.78 GB document in a worker holding 6.6 GB resident.

### 1.4 Re-encoding (`--encode`)

Off by default. When enabled per format, the payload is decoded and rewritten as
a zlib-compressed BJData document:

| source | attachment | bulk keys moved out |
| --- | --- | --- |
| `.nii .nii.gz .hdr .img .img.gz` | `.bnii` | `NIFTIData`, `NIFTIExtension` |
| `.snirf` | `.bnirs` | `SNIRFData` |
| `.gii` | `.bgii` | `GIFTIData` |
| `.jmsh` | `.bmsh` | all |
| `.mat .set` | `.jdb` | all |

Named `<sha256>_<codec>.<ext>` from the **source** file's digest, not the
encoded output's, so the name survives a change of encoder or compression
settings and two dataset versions sharing a file share one attachment.

zlib is the codec because it is native to MATLAB, Python, R, Julia and every
browser. `jdata.zlibmt` removes its speed penalty by deflating independent
`Z_FULL_FLUSH` blocks concurrently and concatenating them into one ordinary zlib
stream: 11.3x at 32 threads for 0.019% more bytes, byte-identical regardless of
thread count.

### 1.5 Content hashes

Taken from the git-annex key whenever one is available — an `MD5E` or `SHA256E`
key already states the content hash and the exact size, and `os.link()` is a
metadata operation, so registering such a file costs no payload I/O at all.
Any backend whose key embeds a hash is accepted, not only the one matching the
store's own algorithm; insisting on one meant every file under the other backend
was read from disk to recompute a hash already present in its symlink target.

A file with no usable key is hashed with the store's algorithm. Re-encoded
attachments are always named by a sha256 of the source, computed while reading
the payload that is about to be re-encoded anyway.

So one document legitimately holds digests from more than one algorithm. Every
`_DataLink_` and every manifest line therefore names its own.

## 2. Output on disk

One current document per dataset. CouchDB produces a hash for every revision it
stores, so it is already the version authority; a parallel per-version archive
here would be a second versioning scheme to keep in step with it.

```
<output>/<ds>/doc.json          the digest
<output>/<ds>/manifest.tsv      <algo>:<hash> <TAB> <size> <TAB> <relpath>, sorted by path
<output>/<ds>/derivatives.json  the split-out subtree, if any
<output>/<ds>/meta.json         version block, stats, per-file errors
<output>/<ds>/datacite.json     written by `njcli doi`
```

`doc.json` is serialised with sorted keys, `separators=(",",":")` and
`ensure_ascii`, and non-finite floats become `null` (BIDS encodes missing values
as `n/a`, which readers surface as NaN, and NaN is neither valid JSON nor
accepted by Postgres `jsonb`).

`manifest.tsv` is the mapping back to git-annex: every payload is named by the
same content hash its annex key carries, so a consumer holding the manifest can
retrieve exactly the bytes the digest describes.

## 3. Document schema

```jsonc
{
  ".neurojson": { ... },                  // metadata block, see below
  "dataset_description.json": { ... },     // parsed sidecar
  "README": "…", "CHANGES": "…",           // text files as strings
  "participants.tsv": {                    // column-oriented, always inline
    "participant_id": ["sub-01", …],
    "age": [26, 24, …]
  },
  "sub-01": {
    "anat": {
      "sub-01_T1w.nii.gz": {
        "NIFTIHeader": { "Dim": [160,192,192], "VoxelSize": [1,1.333,1.333], … },
        "NIFTIData":   { "_DataLink_": "…" }
      }
    },
    "func": {
      "sub-01_task-x_events.tsv": { "onset": [...], "duration": [...] }
    }
  },
  "derivatives": { "_DataLink_": "couch:<db>_derivative/<ds>" }
}
```

### 3.1 The `.neurojson` block

| field | meaning |
| --- | --- |
| `Version` | the upstream release this digest **is**, or `null` when it is not one. Set only when HEAD sits exactly on a semver tag, so filtering on it returns citable snapshots and nothing else. |
| `VersionExact` | whether `Version` is set |
| `BaseVersion` | the release it descends from |
| `CommitsAhead` | how far HEAD has moved past that release |
| `VersionLabel` | `git describe` handle — `1.0.0`, `1.0.0+1.g772e8447`, `0.0.0+g2cee6d75`. For legibility only; semver ignores build metadata for precedence, so it identifies nothing on its own. |
| `VersionSource` | `git-tag`, `git-tag+commits`, `dataset_description.DatasetDOI+commit`, `git-commit`, `none` |
| `SourceCommit`, `SourceRemote` | ground truth for retrieval |
| `Tags` | all semver tags in the checkout |
| `Files`, `Bytes` | totals over the manifest |
| `HashAlgorithm`, `HashSource` | which scheme produced the identifiers |
| `Encoded`, `EncodeCodec` | present only when `--encode` was used |
| `CreateTime`, `UpdateTime` | **set by the server**, never by the client |

There is deliberately no self-computed content fingerprint. The document carries
*upstream* identifiers; CouchDB's `_rev` is the version.

Why `Version` may be null: a dataset tagged `1.0.0` and then updated without a
new tag still declares `.v1.0.0` in its `DatasetDOI`, because maintainers rarely
revise that field. Trusting the declaration would label newer content `1.0.0`.
Across the 1548-dataset mirror, 62 datasets have HEAD ahead of their newest tag
and 30 of those still declare the superseded version.

### 3.2 `_DataLink_`

```
https://neurojson.io/io/cas.cgi?action=get
   &db=<db>&doc=<ds>
   &hash=<algo>:<hex>
   [&enc=_<codec>.<ext>]
   &size=<bytes>
   &file=<relpath>[:$.<JSONPath>]
```

Only `hash` is authoritative. `db`/`doc` keep `jdata.jfile.jsoncache`'s download
cache layout working, `size` is shown before downloading, and `file` **must stay
last** because the `links` view's filename regex is anchored at end of string.
`enc` names a derived encoding stored beside the source content. A trailing
`:$.KEY` addresses one key inside a binary JData attachment.

Two non-URL forms appear: `couch:<db>/<ds>` for the cross-database reference to a
split-out subtree, and `unreadable:<path>` for a file that could not be read.

## 4. Publishing

`POST <db>/_design/qq/_update/replace/<docid>` — never a PUT for data documents.
The handler owns `_rev` resolution and the timestamps, which is what lets the
converter's output stay a pure function of the dataset.

Documents are converted whole and trimmed only when the server refuses one. A
JSON byte count is the wrong budget: CouchDB limits the *internal* size of a
parsed document, and the ratio to JSON depends entirely on content. Measured
against CouchDB 3.4.2 with an 8 MB limit, the largest JSON accepted was

| shape | accepted |
| --- | --- |
| one big string | 4.19 MB |
| many short keys | 7.23 MB |
| float array | ≥29.4 MB |

A sevenfold spread. On refusal, the publish step sheds the largest top-level
subtree into the store, rewrites the document on disk so the archive matches
what was published, and retries; the smallest refused size is remembered so
later documents are trimmed before the attempt rather than after. Dataset-level
metadata is never shed.

An oversized PUT returns a clean `413 document_too_large`; an oversized POST to
an update handler just closes the socket, so a transport failure counts as
too-large provided the server still answers.

## 5. Changes to the design document

Against `neurojson/design/frozen`, which is the document as deployed for the
legacy `openneuro` database.

### Unchanged
`view_subjects.js`, `view_participantsfields.js`, `validate_doc_update.js`.

`subjects` matters most: the Postgres search layer keeps a hand-written port of
it in `backend/sync/incrementalSync.js`, whose own comment warns the copies
"drift silently". Verified identical, row for row and value for value.

### `update_replace.js` — new, and now the publish path
`timestamp` merges the request into the stored document. That is right for a
partial update such as adding an `AISummary`, but wrong for a digest: verified
live, a subject removed upstream survived in the published document
indefinitely, leaving the `subjects` view emitting rows for data that no longer
existed and `Files` disagreeing with the document it described.

`replace` rebuilds the body from the request, preserving only `_id`, `_rev`,
CouchDB's own underscore fields (notably `_attachments`) and `CreateTime`. Both
handlers are installed.

### `update_timestamp.js` — rewritten
Metadata key `.datainfo` → `.neurojson`. `CreateTime` comes from the server
clock instead of the hardcoded `1706237400.000`, and a client-supplied
`CreateTime`/`UpdateTime` is ignored rather than merged in.

### `view_links.js` — content hash added
Keeps the id-first key `[doc._id, ext, size]` that the Postgres sync
range-queries with (fetching the whole links view of a large database exceeds
Node's ~512 MB maximum string size). The value gains `algo` and `hash`, parsed
out of the URL, so a link row identifies immutable content.

### `view_dbinfo.js` — version metadata added
`.datainfo` → `.neurojson`, and the emitted value gains `version`, `label`,
`exact`, `baseversion`, `commitsahead`, `commit`, `remote`, `files`, `bytes`.

**Needs a matching one-line change in the search backend.** `transformDbinfo` in
`incrementalSync.js` reads only `doc["dataset_description.json"]`, so these
fields never reach `ioviews` on an incremental sync. `sync/refreshDbinfo.js`
re-reads the live view and fills them in without any code change.

### `view_versions.js` — new
One row per published digest, keyed `[doc._id, VersionLabel]`, carrying the
version block, commit and totals. Lets the on-disk output be reconciled against
what is actually published.

### `view_updatetime.js` — key rename only
`.datainfo` → `.neurojson`. Note both timestamps are now numbers; the deployed
handler wrote `CreateTime` as a number and `UpdateTime` as a string.

## 6. Compatibility with the search layer

Three things the document must keep, because `incrementalSync.js` ports the map
functions locally and would drift silently otherwise:

1. top-level shape — `dataset_description.json`, `README`/`.md`/`.rst`, `sub-*`
   keys, and `participants.tsv` as column-oriented arrays;
2. the `links` key shape `[doc._id, ext, size]`, with an `ext` that survives
   `isValidFileType()` (dot-prefixed, no slash, ≤20 characters);
3. registration in `sys/registry` — a database absent from it is never indexed,
   however many documents it holds.

`neurojson/pgcompat.js` runs the backend's own ports against a converted
document so drift is caught by `test/testnjviews.py` rather than in production.
