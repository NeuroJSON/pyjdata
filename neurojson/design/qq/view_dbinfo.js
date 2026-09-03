function (doc) {
  log("processing bids" + doc._id);
  var txt = (doc['README'] ? doc['README'] : (doc['README.md'] ? doc['README.md'] : (doc['README.rst'] ? doc['README.rst'] : '')));
  var rawtext = JSON.stringify(doc);
  var datainfo = doc['dataset_description.json'] ? doc['dataset_description.json'] : {Name: doc._id, }
  if (doc['.neurojson']) {
      datainfo['.neurojson'] = doc['.neurojson'];
  }
  var topitems = Object.keys(doc);
  var subjlist = [];
  var modalitylist = [];
  for(var i=0;i<topitems.length;i++) {
    var item = topitems[i];
    if(item.indexOf('ub-') !== -1) {
      subjlist.push(item);
      var modlist=Object.keys(doc[item]);
      for(var j=0;j<modlist.length;j++) {
        var modal = modlist[j];
        if(modal.indexOf('ses') === 0) {
          var modname = Object.keys(doc[item][modal]);
          for(var k=0;k<modname.length;k++) {
            if(modname[k].indexOf('.') === -1 && modalitylist.indexOf(modname[k]) === -1) {
              modalitylist.push(modname[k]);
            }
          }
        } else if(modal.indexOf('.') === -1 && modalitylist.indexOf(modal) === -1) {
            modalitylist.push(modal);
        }
      }
    }
  }
  if(subjlist.length==0) {
      subjlist = ['nonbids'];
  }
  if(modalitylist.length==0) {
      if(rawtext.indexOf('"MeshNode"') !== -1)
          modalitylist.push('JMesh');
      if(rawtext.indexOf('"NIFTIData"') !== -1)
          modalitylist.push('JNIFTI');
      if(rawtext.indexOf('"SNIRFData"') !== -1)
          modalitylist.push('JSNIRF');
      if(rawtext.indexOf('"_ArrayType_"') !== -1)
          modalitylist.push('JData');
  }
  var nj = doc['.neurojson'] || {};
  emit(doc._id, {name: datainfo.Name, length: rawtext.length, readme: txt.substr(0, 256), info: datainfo, subj: subjlist, modality: modalitylist, aisummary: nj.AISummary ? nj.AISummary : '', version: nj.Version ? nj.Version : '', label: nj.VersionLabel ? nj.VersionLabel : (nj.Version ? nj.Version : ''), exact: nj.VersionExact ? true : false, baseversion: nj.BaseVersion ? nj.BaseVersion : '', commitsahead: (nj.CommitsAhead === 0 || nj.CommitsAhead) ? nj.CommitsAhead : null, commit: nj.SourceCommit ? nj.SourceCommit : '', remote: nj.SourceRemote ? nj.SourceRemote : '', files: nj.Files ? nj.Files : 0, bytes: nj.Bytes ? nj.Bytes : 0});
}