function (doc) {
  if(doc['.neurojson'] && doc['.neurojson']['UpdateTime']) {
    emit(doc['.neurojson']['UpdateTime'], doc['.neurojson']['CreateTime']);
  }
}
