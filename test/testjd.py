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
data['a_biginteger']=9007199254740991;
data['a_map']={
  float('nan'): 'one',
  2: 'two',
  "k": 'three'
};


def run():
    print('== Original Python native data ==')
    newdata=data.copy();
    print(newdata);

    print('== JData-annotated data ==')
    print(jd.show(jd.encode(newdata),indent=4, default=jd.jsonfilter));
    
    print('== JData-annotated data exported to JSON with zlib compression ==')
    newdata=data.copy();
    print(jd.show(jd.encode(newdata,{'compression':'zlib','base64':True}), indent=4, default=jd.jsonfilter));

    print('== Decoding a JData-encoded data and printed in JSON format ==')
    newdata=data.copy();
    print(jd.show(jd.decode(jd.encode(newdata)), indent=4, default=jd.jsonfilter));

    print('== Saving encoded data to test.json ==')
    jd.save(data,'test.json')
    
    print('== Loading data from test.json and decode ==')
    newdata=jd.load('test.json')
    print(jd.show(newdata, indent=4, default=jd.jsonfilter));
    