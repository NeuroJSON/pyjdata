function (doc) {
  log("processing " + doc._id);
  const filename = /file=([^\/]*\/)*([^&\/]+?)(\.[^.&%:]+(\.gz)*)([&:].*)*$/;
  const filesize = /size=(\d+)/;
  const jsonpath = /:(\$[^&]+)/;
  const dotpat = /\./g;
  let url, fname, fsize, jpath, uniqurl, urlhash = {};
  function traverse(obj, level, rootpath) {
      if(level > 10)
          return;
      Object.keys(obj).forEach(function (subkey) {
          if(subkey == '_DataLink_' && typeof obj[subkey] === 'string' && obj[subkey].indexOf('http')!== -1) {
              url = obj[subkey];
              uniqurl = url.split(':$')[0];
              if(!urlhash.hasOwnProperty(uniqurl)) {
                fname = uniqurl.match(filename);
                if(fname === null) {
                  log("no suffix:" + url);
                  return;
                }
                fsize = url.match(filesize);
                var size = fsize === null ? 0 : parseInt(fsize[1]);
                jpath = url.match(jsonpath);
                if(jpath !== null && jpath.length)
                    jpath = jpath[1];
                urlhash[uniqurl] = 1;
                emit([doc._id, fname[3], size], {path: rootpath, url: uniqurl, file: fname[2] + fname[3], suffix: fname[3], ref: jpath});
              }
          }
          if(obj[subkey] !== null && typeof obj[subkey] === 'object') {
              traverse(obj[subkey], level+1, rootpath + '.' + subkey.replace(dotpat, '\\.'));
          }
      });
  }
  traverse(doc, 1, '$');
}