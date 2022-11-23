![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# JData for Python - lightweight and serializable data annotations for Python

- Copyright: (C) Qianqian Fang (2019-2022) <q.fang at neu.edu>
- License: Apache License, Version 2.0
- Version: 0.5.2
- URL: https://github.com/NeuroJSON/pyjdata

[![Build Status](https://travis-ci.com/fangq/pyjdata.svg?branch=master)](https://travis-ci.com/fangq/pyjdata)

The [JData Specification](https://github.com/NeuroJSON/jdata/) defines a lightweight 
language-independent data annotation interface targetted at
storing and sharing complex data structures across different programming
languages such as MATLAB, JavaScript, Python etc. Using JData formats, a 
complex Python data structure can be encoded as a `dict` object that is easily 
serialized as a JSON/binary JSON file and share such data between
programs of different languages.

## How to install

* Github: download from https://github.com/NeuroJSON/pyjdata
* PIP: run `pip install jdata` see https://pypi.org/project/jdata/

This package can also be installed on Ubuntu 21.04 or Debian Bullseye via
```
sudo apt-get install python3-jdata
```

On older Ubuntu or Debian releases, you may install jdata via the below PPA:
```
sudo add-apt-repository ppa:fangq/ppa
sudo apt-get update
sudo apt-get install python3-jdata
```

Dependencies:
* **numpy**: PIP: run `pip install numpy` or `sudo apt-get install python3-numpy`
* (optional) **bjdata**: PIP: run `pip install bjdata` or `sudo apt-get install python3-bjdata`, see https://pypi.org/project/bjdata/, only needed to read/write BJData/UBJSON files
* (optional) **lz4**: PIP: run `pip install lz4`, only needed when encoding/decoding lz4-compressed data
* (optional) **backports.lzma**: PIP: run `sudo apt-get install liblzma-dev` and `pip install backports.lzma` (needed for Python 2.7), only needed when encoding/decoding lzma-compressed data
* (optional) **blosc2**: PIP: run `pip install blosc2`, only needed when encoding/decoding blosc2-compressed data

Replacing `pip` by `pip3` if you are using Python 3.x. If either `pip` or `pip3` 
does not exist on your system, please run
```
    sudo apt-get install python3-pip
```
Please note that in some OS releases (such as Ubuntu 20.04), python2.x and python-pip 
are no longer supported.

One can also install this module from the source code. To do this, you first
check out a copy of the latest code from Github by
```
    git clone https://github.com/NeuroJSON/pyjdata.git
    cd pyjdata
```
then install the module to your local user folder by
```
    python3 setup.py install --user
```
or, if you prefer, install to the system folder for all users by
```
    sudo python3 setup.py install
```
Please replace `python` by `python3` if you want to install it for Python 3.x instead of 2.x.

Instead of installing the module, you can also import the jdata module directly from 
your local copy by cd the root folder of the unzipped pyjdata package, and run
```
   import jdata as jd
```

## How to use

The PyJData module is easy to use. You can use the `encode()/decode()` functions to
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

PyJData supports multiple N-D array data compression/decompression methods (i.e. codecs), similar
to HDF5 filters. Currently supported codecs include `zlib`, `gzip`, `lz4`, `lzma`, `base64` and various
`blosc2` compression methods, including `blosc2blosclz`, `blosc2lz4`, `blosc2lz4hc`, `blosc2zlib`,
`blosc2zstd`. To apply a selected compression method, one simply set `{'compression':'method'}` as
the option to `jdata.encode` or `jdata.save` function; `jdata.load` or `jdata.decode` automatically
decompress the data based on the `_ArrayZipType_` annotation present in the data. Only `blosc2`
compression methods support multi-threading. To set the thread number, one should define an `nthread`
value in the option (`opt`) for both encoding and decoding.


## Utility

One can convert from JSON based data files (`.json, .jdt, .jnii, .jmsh, .jnirs`) to binary-JData
based binary files (`.bjd, .jdb, .bnii, .bmsh, .bnirs`) and vice versa using command
```
python3 -mjdata /path/to/text/json/file.json # convert to /path/to/text/json/file.jdb
python3 -mjdata /path/to/text/json/file.jdb  # convert to /path/to/text/json/file.json
python3 -mjdata -h                           # show help info
```

## Test

To see additional data type support, please run the built-in test using below command

```
python3 -m unittest discover -v test
```
