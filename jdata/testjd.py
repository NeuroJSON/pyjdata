"""jdata.py test unit

To run the test, please run

   import testjd
   testjd.run()

Copyright (c) 2019 Qianqian Fang <q.fang at neu.edu>
"""

import jdata as jd
import numpy as np
import collections
import json

data=collections.OrderedDict();
data['const']=[2.0, 1, True, False, None, float('nan'), float('-inf')];
data['shortarray']=[1,2,3];
data['a_complex']=1+2.0j;
data['object']=[[[1],[2],[3]],None, False];
data['a_typedarray']=np.asarray([9,9,9,9],dtype=np.uint8);
data['a_ndarray']=np.arange(1,10,dtype=np.int32).reshape(3,3);
data['a_biginteger']=long(9007199254740991);
data['a_map']={
  float('nan'): 'one',
  2: 'two',
  "k": 'three'
};

def _exportfilter(o):
    if isinstance(o, long): return str(o) 

def run():
    #newdata=data.copy();
    #print(jd.encode(newdata));
    #newdata=data.copy();
    #print(json.dumps(jd.encode(newdata,{'compression':'zlib'}), indent=4, default=_exportfilter));
    newdata=data.copy();
    print(json.dumps(jd.decode(jd.encode(newdata)), indent=4, default=_exportfilter));
