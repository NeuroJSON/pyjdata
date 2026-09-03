#!/usr/bin/env node
// Run CouchDB design-document JavaScript outside CouchDB.
//
// CouchDB views are JavaScript evaluated by the server, so a mistake in a map
// function is only discovered after the document is published and the index
// rebuilt.  This harness evaluates the same source files against a local
// document so view output can be asserted in a test, and it also drives the
// update handler through a sequence of pushes to check the timestamp semantics.
//
//   node simulate.js view   <designdir> <view> <doc.json> [doc.json ...]
//   node simulate.js update <designdir> <handler> <doc.json> [doc.json ...]
//
// "view" prints one JSON object per emitted row: {id, key, value}.
// "update" applies the handler repeatedly, printing the resulting metadata
// block after each application so CreateTime/UpdateTime behaviour is visible.

const fs = require("fs");
const path = require("path");

// piping into head/less closes stdout early; that is not an error here
process.stdout.on("error", (e) => {
  if (e.code === "EPIPE") process.exit(0);
  throw e;
});

// A digest is stored as <ds>/<version>/doc.json, so the document id is the
// dataset directory rather than the file name.
function docId(f, doc) {
  if (doc._id) return doc._id;
  const base = path.basename(f).replace(/\.json$/, "");
  if (base === "doc" || base === "derivatives") {
    return path.basename(path.dirname(path.dirname(path.resolve(f))));
  }
  return base;
}

function loadFn(dir, file) {
  const src = fs.readFileSync(path.join(dir, file), "utf8");
  return eval("(" + src + ")");
}

const [, , mode, designdir, name, ...docfiles] = process.argv;
if (!mode || !designdir || !name || docfiles.length === 0) {
  console.error("usage: simulate.js view|update <designdir> <name> <doc.json>...");
  process.exit(2);
}

// CouchDB exposes log()/sum()/toJSON() to view code
global.log = function () {};
global.sum = (a) => a.reduce((x, y) => x + y, 0);
global.toJSON = JSON.stringify;

if (mode === "view") {
  const map = loadFn(designdir, "view_" + name + ".js");
  for (const f of docfiles) {
    const doc = JSON.parse(fs.readFileSync(f, "utf8"));
    doc._id = docId(f, doc);
    global.emit = (key, value) =>
      process.stdout.write(JSON.stringify({ id: doc._id, key, value }) + "\n");
    map(doc);
  }
} else if (mode === "update") {
  const handler = loadFn(designdir, "update_" + name + ".js");
  for (const f of docfiles) {
    const body = fs.readFileSync(f, "utf8");
    const parsed = JSON.parse(body);
    const docid = docId(f, parsed);
    let stored = null;
    // three consecutive pushes of the same payload: insert, then two updates
    for (let i = 0; i < 3; i++) {
      const req = { body, id: docid, uuid: docid };
      const out = handler(stored, req);
      stored = out[0];
      process.stdout.write(
        JSON.stringify({
          push: i + 1,
          id: stored._id,
          meta: stored[".neurojson"],
          topkeys: Object.keys(stored).slice(0, 6),
        }) + "\n"
      );
    }
  }
} else {
  console.error("unknown mode " + mode);
  process.exit(2);
}
