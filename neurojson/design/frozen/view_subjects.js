function (doc) {
  log("processing " + doc._id);
  var skipkey=0;
  ["sidecards", "derivatives", "sourcedata", "code"].every(function (foldername) {
    if(doc._id === foldername) {
      skipkey = 1;
      return false;
    }
    return true;
  });
  if(skipkey === 1) {
    return;
  }

  Object.keys(doc).forEach(function (subj) {
     if(subj.match(/^[sS]ub-/)) {
      var sessionlist = [];
      var modalitylist = [];
      var tasklist = [];
      var runlist = [];
      var filetype = [];
      var age=-0.01;
      var gender='N';
      if (doc['participants.tsv'] && doc['participants.tsv']['participant_id'] && Array.isArray(doc['participants.tsv']['participant_id'])) {
        var idx = -1;
        for(var i=0; i< doc['participants.tsv'].participant_id.length; i++) {
          if(subj.indexOf(doc['participants.tsv'].participant_id[i].toString()) > -1) {
            idx = i;
            break;
          }
        }
        if(idx >= 0) {
          ["age","age_scan","age_at_scan"].every(function (agekey) {
            if(age < 0) {
              if(doc['participants.tsv'][agekey]) {
                age = doc['participants.tsv'][agekey][idx];
                return false;
              } else if(doc['participants.tsv'][agekey.toUpperCase()]) {
                age = doc['participants.tsv'][agekey.toUpperCase()][idx];
                return false;
              } else if(doc['participants.tsv'][agekey.charAt(0).toUpperCase() + agekey.slice(1)]) {
                age = doc['participants.tsv'][agekey.charAt(0).toUpperCase() + agekey.slice(1)];
                return false;
              }
            }
            return true;
          });

          if(age < 0) {
              Object.keys(doc['participants.tsv']).forEach(function (pfield) {
                if(pfield.toLowerCase().indexOf('age') >= 0)
                  age = doc['participants.tsv'][pfield][idx];
              });
          }
          ["sex","gender"].every(function (sexkey) {
            if(gender === 'N') {
              if(doc['participants.tsv'][sexkey]) {
                gender = doc['participants.tsv'][sexkey][idx];
                return false;
              } else if(doc['participants.tsv'][sexkey.toUpperCase()]) {
                gender = doc['participants.tsv'][sexkey.toUpperCase()][idx];
                return false;
              } else if(doc['participants.tsv'][sexkey.charAt(0).toUpperCase() + sexkey.slice(1)]) {
                gender = doc['participants.tsv'][sexkey.charAt(0).toUpperCase() + sexkey.slice(1)];
                return false;
              }
            }
            return true;
          });
          if(gender === 'N' ) {
              Object.keys(doc['participants.tsv']).forEach(function (pfield) {
                if(pfield.toLowerCase().indexOf('sex') >= 0)
                  gender = doc['participants.tsv'][pfield][idx];
              });
          }
          if(gender === 'N' ) {
              Object.keys(doc['participants.tsv']).forEach(function (pfield) {
                if(pfield.toLowerCase().indexOf('gender') >= 0)
                  gender = doc['participants.tsv'][pfield][idx];
              });
          }
        }
      }
      Object.keys(doc[subj]).forEach(function (modal) {
        if(modal.indexOf('ses-') === 0) {
          if(sessionlist.indexOf(modal.substring(4)) === -1) {
            sessionlist.push(modal.substring(4));
          }
          Object.keys(doc[subj][modal]).forEach(function (modname) {
            if(modname.indexOf('.') === -1 && modalitylist.indexOf(modname) == -1) {
              modalitylist.push(modname);
            }
            Object.keys(doc[subj][modal][modname]).forEach(function (filename) {
              filename.split('_').forEach( function (task) {
                  if(task.indexOf('run-')===0) {
                    if(runlist.indexOf(task.substring(4)) === -1) {
                      runlist.push(task.substring(4));
                    }
                  } else if(task.indexOf('task-') === 0) {
                    if(tasklist.indexOf(task.substring(5)) === -1) {
                      tasklist.push(task.substring(5));
                    }
                  } else if(task.indexOf('.') > 0) {
                    var tmp = task.substring(0,task.indexOf('.'));
                    if(filetype.indexOf(tmp) === -1) {
                      filetype.push(tmp);
                    }
                  }
              });
            });
          });
        } else if(modal.indexOf('.') === -1 && modalitylist.indexOf(modal) === -1) {
            modalitylist.push(modal);
            Object.keys(doc[subj][modal]).forEach(function (filename) {
              filename.split('_').forEach( function (task) {
                  if(task.indexOf('run-')===0) {
                    if(runlist.indexOf(task.substring(4)) === -1) {
                      runlist.push(task.substring(4));
                    }
                  } else if(task.indexOf('task-') === 0) {
                    if(tasklist.indexOf(task.substring(5)) === -1) {
                      tasklist.push(task.substring(5));
                    }
                  } else if(task.indexOf('.') > 0) {
                    var tmp = task.substring(0,task.indexOf('.'));
                    if(filetype.indexOf(tmp) === -1) {
                      filetype.push(tmp);
                    }
                  }
              });
            });
        }
      });
      if(typeof gender === 'string' ) {
        gender=gender.substring(0,1).toUpperCase();
      } else {
        gender=gender+'';
      }
      if(typeof age === 'string' && isNaN(+age)) {
        age=-0.001;
      }
      if(typeof age === 'string') {
        age=+age;
      }
      if(typeof age < 0) {
        age=-0.01;
      }
      age=Math.floor(age*100);
      emit([('0000' + age).slice(-5), ('000' + gender).slice(-4),('000' +sessionlist.length).slice(-4),('000' + modalitylist.length).slice(-4), ('000' + tasklist.length).slice(-4), ('000' + runlist.length).slice(-4), subj.substring(4)], {'sessions': sessionlist, 'modalities': modalitylist, 'tasks': tasklist, 'runs': runlist, 'types': filetype});
    }
  });
}