function (doc) {
  if(doc['.neurojson']) {
    var nj = doc['.neurojson'];
    // Keyed by the human label; Version is set only for an exact upstream
    // release, so it is reported separately.  No content hash of our own: the
    // revision CouchDB assigns is the version, and the identifiers here are the
    // upstream ones that map a digest back to git-annex.
    emit([doc._id, nj.VersionLabel || nj.Version || ''], {
      version: nj.Version || '',
      label: nj.VersionLabel || nj.Version || '',
      exact: nj.VersionExact ? true : false,
      baseversion: nj.BaseVersion || '',
      commitsahead: (nj.CommitsAhead === 0 || nj.CommitsAhead) ? nj.CommitsAhead : null,
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
