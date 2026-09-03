function (doc) {
  if(doc['participants.tsv']) {
    emit(Object.keys(doc['participants.tsv']), 1);
  }
}