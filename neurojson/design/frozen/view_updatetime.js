function (doc) {
  if(doc['.datainfo'] && doc['.datainfo']['UpdateTime']) {
    emit(doc['.datainfo']['UpdateTime'], doc['.datainfo']['CreateTime']);
  }
}