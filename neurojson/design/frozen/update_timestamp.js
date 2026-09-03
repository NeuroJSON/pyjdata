function(doc, req) {
  var body = JSON.parse(req.body || '{}');
  var metakey = '.datainfo';
  if (!doc || !doc.hasOwnProperty(metakey)) {
    if(!doc) {
        doc = { _id: req.id || req.uuid};
        doc[metakey] = {};
    } else {
        var newdoc = {_id: doc._id};
        newdoc[metakey] = {};
        for (var key in doc) {
            newdoc[key] = doc[key];
        }
        doc = newdoc;
    }
    doc[metakey].CreateTime = 1706237400.000; //db.create_time //(Date.now()/1000).toString()
  }

  doc[metakey].UpdateTime = (Date.now()/1000).toString();

  for (var key in body) {
    if(key === metakey && doc.hasOwnProperty(key)) {
        for (var infokey in body[metakey]) {
            doc[metakey][infokey] = body[metakey][infokey];
        }
    } else {
        doc[key] = body[key];
    }
  }
  return [doc, 'Document updated successfully'];
}