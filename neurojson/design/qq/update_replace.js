function(doc, req) {
  // Publish a digest as a *complete* replacement of the document body.
  //
  // The `timestamp` handler merges the request into the existing document,
  // which is right for a partial update (adding an AI summary, say) but wrong
  // for publishing a digest: a subject or file removed upstream would survive
  // in the published document forever. That leaves the subjects view emitting
  // rows for data that no longer exists, .neurojson.Files disagreeing with the
  // document it describes, and -- worst -- the fingerprint no longer describing
  // the bytes that were published, which is the one thing it exists to do.
  //
  // Preserved from the stored document: _id, _rev, CouchDB's own underscore
  // fields (notably _attachments), and CreateTime. Everything else comes from
  // the request. CreateTime and UpdateTime are the server's to set, so a value
  // supplied by a client is ignored.
  var body = JSON.parse(req.body || '{}');
  var metakey = '.neurojson';
  var now = Date.now() / 1000;
  var created = now;
  var out = {};

  if (doc) {
    out._id = doc._id;
    if (doc._rev) { out._rev = doc._rev; }
    for (var old in doc) {
      // keep CouchDB's internal fields, drop the previous content
      if (old.charAt(0) === '_' && old !== '_id' && old !== '_rev') {
        out[old] = doc[old];
      }
    }
    if (doc[metakey] && doc[metakey].CreateTime) {
      created = doc[metakey].CreateTime;
    }
  } else {
    out._id = req.id || req.uuid;
  }

  out[metakey] = {};
  if (body[metakey]) {
    for (var info in body[metakey]) {
      if (info !== 'CreateTime' && info !== 'UpdateTime') {
        out[metakey][info] = body[metakey][info];
      }
    }
  }
  out[metakey].CreateTime = created;
  out[metakey].UpdateTime = now;

  for (var key in body) {
    if (key !== metakey && key !== '_id' && key !== '_rev') {
      out[key] = body[key];
    }
  }
  return [out, JSON.stringify({ ok: true, id: out._id, updated: now }) + '\n'];
}
