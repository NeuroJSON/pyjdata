function (doc) {
  if(doc['.neurojson'] && doc['.neurojson'].Fingerprint) {
    var nj = doc['.neurojson'];
    emit([doc._id, nj.Version || ''], {
      fingerprint: nj.Fingerprint,
      commit: nj.SourceCommit || '',
      versionsource: nj.VersionSource || '',
      tags: nj.Tags || [],
      files: nj.Files || 0,
      bytes: nj.Bytes || 0,
      createtime: nj.CreateTime || null,
      updatetime: nj.UpdateTime || null
    });
  }
}
