![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# JData for Python - lightweight and serializable data annotations for Python

- Copyright: (C) Qianqian Fang (2019-2024) <q.fang at neu.edu>
- License: Apache License, Version 2.0
- Version: 0.6.0
- URL: https://github.com/NeuroJSON/pyjdata
- Acknowledgement: This project is supported by US National Institute of Health (NIH)
  grant [U24-NS124027](https://reporter.nih.gov/project-details/10308329)

![Build Status](https://github.com/NeuroJSON/pyjdata/actions/workflows/run_test.yml/badge.svg)

The [JData Specification](https://github.com/NeuroJSON/jdata/) defines a lightweight 
language-independent data annotation interface enabling easy storing
and sharing of complex data structures across different programming
languages such as MATLAB, JavaScript, Python etc. Using JData formats, a 
complex Python data structure, including numpy objects, can be encoded
as a simple `dict` object that is easily serialized as a JSON/binary JSON
file and share such data between programs of different languages.

Since 2021, the development of PyJData module and the underlying data format specificaitons
[JData](https://neurojson.org/jdata/draft3) and [BJData](https://neurojson.org/bjdata/draft2)
have been funded by the US National Institute of Health (NIH) as
part of the NeuroJSON project (https://neurojson.org and https://neurojson.io).

The goal of the NeuroJSON project is to develop scalable, searchable, and
reusable neuroimaging data formats and data sharing platforms. All data
produced from the NeuroJSON project will be using JSON/Binary JData formats as the
underlying serialization standards and the lightweight JData specification as
language-independent data annotation standard.

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

One can use `loadt` or `savet` to read/write JSON-based data files and `loadb` and `saveb` to
read/write binary-JSON based data files. By default, JData annotations are automatically decoded
after loading and encoded before saving. One can set `{'encode': False}` in the save functions
or `{'decode': False}` in the load functions as the `opt` to disable further processing of JData
annotations. We also provide `loadts` and `loadbs` for parsing a string-buffer made of text-based
JSON or binary JSON stream.

PyJData supports multiple N-D array data compression/decompression methods (i.e. codecs), similar
to HDF5 filters. Currently supported codecs include `zlib`, `gzip`, `lz4`, `lzma`, `base64` and various
`blosc2` compression methods, including `blosc2blosclz`, `blosc2lz4`, `blosc2lz4hc`, `blosc2zlib`,
`blosc2zstd`. To apply a selected compression method, one simply set `{'compression':'method'}` as
the option to `jdata.encode` or `jdata.save` function; `jdata.load` or `jdata.decode` automatically
decompress the data based on the `_ArrayZipType_` annotation present in the data. Only `blosc2`
compression methods support multi-threading. To set the thread number, one should define an `nthread`
value in the option (`opt`) for both encoding and decoding.

## Reading JSON via REST-API

If a REST-API (URL) is given as the first input of `load`, it reads the JSON data directly
from the URL and parse the content to native Python data structures. To avoid repetitive download,
`load` automatically cache the downloaded file so that future calls directly load the
locally cached file. If one prefers to always load from the URL without local cache, one should
use `loadurl()` instead. Here is an example

```
import jdata as jd
data = jd.load('https://neurojson.io:7777/openneuro/ds000001');
data.keys()
```

## Using JSONPath to access and query complex datasets

Starting from v0.6.0, PyJData provides a lightweight implementation [JSONPath](https://goessner.net/articles/JsonPath/),
a widely used format for query and access a hierarchical dict/list structure, such as those
parsed by `load` or `loadurl`. Here is an example

```
import jdata as jd

data = jd.loadurl('https://raw.githubusercontent.com/fangq/jsonlab/master/examples/example1.json');
jd.jsonpath(data, '$.age')
jd.jsonpath(data, '$.address.city')
jd.jsonpath(data, '$.phoneNumber')
jd.jsonpath(data, '$.phoneNumber[0]')
jd.jsonpath(data, '$.phoneNumber[0].type')
jd.jsonpath(data, '$.phoneNumber[-1]')
jd.jsonpath(data, '$.phoneNumber..number')
jd.jsonpath(data, '$[phoneNumber][type]')
jd.jsonpath(data, '$[phoneNumber][type][1]')
```

The `jd.jsonpath` function does not support all JSONPath features. If more complex JSONPath
queries are needed, one should install `jsonpath_ng` or other more advanced JSONPath support.
Here is an example using `jsonpath_ng`

```
import jdata as jd
from jsonpath_ng.ext import parse

data = jd.loadurl('https://raw.githubusercontent.com/fangq/jsonlab/master/examples/example1.json');

val = [match.value for match in parse('$.address.city').find(data)]
val = [match.value for match in parse('$.phoneNumber').find(data)]
```

## Downloading and caching `_DataLink_` referenced external data files

Similarly to [JSONLab](https://github.com/fangq/jsonlab?tab=readme-ov-file#jsoncachem),
PyJData also provides similar external data file downloading/caching capability.

The `_DataLink_` annotation in the JData specification permits linking of external data files
in a JSON file - to make downloading/parsing externally linked data files efficient, such as
processing large neuroimaging datasets hosted on http://neurojson.io, we have developed a system
to download files on-demand and cache those locally. jsoncache.m is responsible of searching
the local cache folders, if found the requested file, it returns the path to the local cache;
if not found, it returns a SHA-256 hash of the URL as the file name, and the possible cache folders

When loading a file from URL, below is the order of cache file search paths, ranking in search order
```
   global-variable NEUROJSON_CACHE | if defined, this path will be searched first
   [pwd '/.neurojson']  	   | on all OSes
   /home/USERNAME/.neurojson	   | on all OSes (per-user)
   /home/USERNAME/.cache/neurojson | if on Linux (per-user)
   /var/cache/neurojson 	   | if on Linux (system wide)
   /home/USERNAME/Library/neurojson| if on MacOS (per-user)
   /Library/neurojson		   | if on MacOS (system wide)
   C:\ProgramData\neurojson	   | if on Windows (system wide)
```
When saving a file from a URL, under the root cache folder, subfolders can be created;
if the URL is one of a standard NeuroJSON.io URLs as below
```
   https://neurojson.org/io/stat.cgi?action=get&db=DBNAME&doc=DOCNAME&file=sub-01/anat/datafile.nii.gz
   https://neurojson.io:7777/DBNAME/DOCNAME
   https://neurojson.io:7777/DBNAME/DOCNAME/datafile.suffix
```
the file datafile.nii.gz will be downloaded to /home/USERNAME/.neurojson/io/DBNAME/DOCNAME/sub-01/anat/ folder
if a URL does not follow the neurojson.io format, the cache folder has the below form
```
   CACHEFOLDER{i}/domainname.com/XX/YY/XXYYZZZZ...
```
where XXYYZZZZ.. is the SHA-256 hash of the full URL, XX is the first two digit, YY is the 3-4 digits

In PyJData, we provide `jdata.jdlink()` function to dynamically download and locally cache
externally linked data files. `jdata.jdlink()` only parse files with JSON/binary JSON suffixes that
`load` supports. Here is a example

```
import jdata as jd

data = jd.load('https://neurojson.io:7777/openneuro/ds000001');
extlinks = jd.jsonpath(data, '$..anat.._DataLink_')  # deep-scan of all anatomical folders and find all linked NIfTI files
jd.jdlink(extlinks, {'regex': 'sub-0[12]_.*nii'})  # download only the nii files for sub-01 and sub-02
jd.jdlink(extlinks)                                # download all links
```

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
