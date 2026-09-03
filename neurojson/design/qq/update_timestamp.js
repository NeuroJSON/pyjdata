function(doc, req) {
  var body = JSON.parse(req.body || '{}');
  var metakey = '.neurojson';
  var now = Date.now() / 1000;
  if (!doc) {
    doc = { _id: req.id || req.uuid };
    doc[metakey] = { CreateTime: now };
  } else if (!doc.hasOwnProperty(metakey)) {
    // preserve key order: the metadata block goes first on an existing doc
    var newdoc = { _id: doc._id, _rev: doc._rev };
    newdoc[metakey] = { CreateTime: now };
    for (var key in doc) {
      if (key !== '_id' && key !== '_rev') { newdoc[key] = doc[key]; }
    }
    doc = newdoc;
  }
  doc[metakey].UpdateTime = now;
  var created = doc[metakey].CreateTime;

  for (var key in body) {
    if (key === metakey) {
      // merge, so a re-push cannot overwrite CreateTime or the server clock
      for (var infokey in body[metakey]) {
        if (infokey !== 'CreateTime' && infokey !== 'UpdateTime') {
          doc[metakey][infokey] = body[metakey][infokey];
        }
      }
    } else if (key !== '_id' && key !== '_rev') {
      doc[key] = body[key];
    }
  }
  doc[metakey].CreateTime = created;
  doc[metakey].UpdateTime = now;
  return [doc, JSON.stringify({ ok: true, id: doc._id, updated: now }) + '\n'];
}
