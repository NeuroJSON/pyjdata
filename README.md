# JData for Python - a lightweight and portable data annotation method

- Copyright: (C) Qianqian Fang (2019) <q.fang at neu.edu>
- License: Apache License, Version 2.0
- Version: 0.2
- URL: https://github.com/fangq/pyjdata


JData Specification is a lightweight data annotation method targetted at
storing and sharing complex data structures between different programming
languages such as MATLAB, JavaScript, Python etc. Using JData format, a 
complex data structure can be encoded as a structure that is easily 
serialized as a JSON/binary JSON file and share such data between
programs of different languages.

The latest version of the JData specification can be found in the file named 
[JData_specification.md](JData_specification.md). The specification is written
in the [Markdown format](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) 
for convenient editing and version control.


## How to use

The PyJData module is easy to use. You can use the encode()/decode() functions to
encode Python data into JData annotation format, or decode JData structures into
native Python data, for example

```
import jdata as jd
import numpy as np
a={'str':'test','num':1.2,'list':[1.1,[2.1]],'nan':float('nan'),'np':np.arange(1,5,dtype=np.uint8)}
jd.encode(a)
jd.decode(jd.encode(a))
d1=jd.encode(a,{'compression':'zlib','base64':1})
d1
jd.decode(d1,{'base64':1})
```

One can further save the JData annotated data into JSON or binary JSON (UBJSON) files using
the `jdata.save` function, or loading JData-formatted data to Python using `jdata.load`

```
import jdata as jd
import numpy as np
a={'str':'test','num':1.2,'list':[1.1,[2.1]],'nan':float('nan'),'np':np.arange(1,5,dtype=np.uint8)}
jd.save(a,'test.json')
newdata=jd.load('test.json')
newdata
```



