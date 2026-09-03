#!/usr/bin/env node
// Check a converted document against the Postgres search layer's expectations.
//
// backend/sync/incrementalSync.js keeps hand-written JavaScript ports of the
// CouchDB dbinfo and subjects map functions, and its own comment warns that
// "if upstream views change, these drift silently".  That makes the ports a
// second, independent consumer of the document schema: a change that is fine
// for CouchDB can still silently empty out Postgres search results.
//
// This script extracts those two functions from the backend source and runs
// them against a document, printing the rows that would be written to the
// `ioviews` table.  Comparing that output with neurojson/simulate.js output for
// the same document is what proves the two sides agree.
//
//   node pgcompat.js <incrementalSync.js> <transform> <doc.json> [docid]
//     transform: dbinfo | subjects

const fs = require("fs");
const path = require("path");

const [, , syncfile, which, docfile, docidArg] = process.argv;
if (!syncfile || !which || !docfile) {
  console.error("usage: pgcompat.js <incrementalSync.js> dbinfo|subjects <doc.json> [docid]");
  process.exit(2);
}

const src = fs.readFileSync(syncfile, "utf8");

// Pull out just the two pure transform functions; the rest of the module opens
// database and network connections at import time.
function extract(name) {
  const start = src.indexOf("function " + name + "(");
  if (start < 0) throw new Error("cannot find function " + name);
  let depth = 0;
  let i = src.indexOf("{", start);
  const from = i;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  return src.slice(start, i + 1);
}

const transformDbinfo = eval("(" + extract("transformDbinfo") + ")");
const transformSubjects = eval("(" + extract("transformSubjects") + ")");

const doc = JSON.parse(fs.readFileSync(docfile, "utf8"));
if (!doc._id) {
  doc._id =
    docidArg ||
    path.basename(path.dirname(path.dirname(path.resolve(docfile))));
}

if (which === "dbinfo") {
  const value = transformDbinfo(doc);
  // firstSync() derives the ioviews.subj column from the subject count
  process.stdout.write(
    JSON.stringify({ id: doc._id, subj: String((value.subj || []).length), view: "dbinfo", value }) + "\n"
  );
} else if (which === "subjects") {
  for (const row of transformSubjects(doc)) {
    process.stdout.write(
      JSON.stringify({ id: row.id, subj: String(row.key[6] || ""), view: "subjects", key: row.key, value: row.value }) + "\n"
    );
  }
} else {
  console.error("unknown transform " + which);
  process.exit(2);
}
