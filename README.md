![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# JData - NeuroJSON client with fast parsers for JSON, binary JSON, NIFTI, SNIRF, CSV/TSV, HDF5 data files

- Copyright: (C) Qianqian Fang (2019-2026) <q.fang at neu.edu>
- License: Apache License, Version 2.0
- Version: 0.9.1
- URL: https://github.com/NeuroJSON/pyjdata
- Acknowledgement: This project is supported by US National Institute of Health (NIH)
  grant [U24-NS124027](https://reporter.nih.gov/project-details/10308329)

![Build Status](https://github.com/NeuroJSON/pyjdata/actions/workflows/run_test.yml/badge.svg)

## Table of Contents

- [Introduction](#introduction)
- [File formats](#file-formats)
- [Submodules](#submodules)
- [How to install](#how-to-install)
- [How to build](#how-to-build)
- [How to use](#how-to-use)
- [Advanced interfaces](#advanced-interfaces)
- [Reading JSON via REST-API](#reading-json-via-rest-api)
- [Using JSONPath to access and query complex datasets](#using-jsonpath-to-access-and-query-complex-datasets)
- [Downloading and caching `_DataLink_` referenced external data files](#downloading-and-caching-_datalink_-referenced-external-data-files)
- [Utility](#utility)
- [How to contribute](#how-to-contribute)
- [Test](#test)

## Introduction

`jdata` is a lightweight and fast neuroimaging data file parser, with built
in support for NIfTI-1/2 (`.nii`, `.nii.gz`), two-part Analyze 7.5 (`.img/.hdr`, `.img.gz`),
HDF5 (`.h5`), SNIRF (`.snirf`), MATLAB .mat files (`.mat`), CSV/TSV (`.csv`, `.csv.gz`,
`.tsv`, `.tsv.gz`), JSON (`.json`), and various binary-JSON data formats, including
BJData (`.bjd`), UBJSON (`.ubj`), and MessagePack (`.msgpack`) formats. `jdata` can
load data files both from local storage and REST-API via URLs. To maximize portability,
the outputs of `jdata` data parsers are intentionally based upon only the **native Python**
data structures (`dict/list/tuple`) plus `numpy` arrays. The entire package is less than
60KB in size and is platform-independent.

`jdata` highly compatible to the [JSONLab toolbox](https://github.com/NeuroJSON/jsonlab)
for MATLAB/Octave, serving as the reference library for Python for the
[JData Specification](https://github.com/NeuroJSON/jdata/),
The JData Specification defines a lightweight
language-independent data annotation interface enabling easy storing
and sharing of complex data structures across different programming
languages such as MATLAB, JavaScript, Python etc. Using JData formats, a 
complex Python data structure, including numpy objects, can be encoded
as a simple `dict` object that is easily serialized as a JSON/binary JSON
file and share such data between programs of different languages.

Since 2021, the development of the `jdata` module and the underlying data format specificaitons
[JData](https://neurojson.org/jdata/draft3) and [BJData](https://neurojson.org/bjdata/draft3)
have been funded by the US National Institute of Health (NIH) as
part of the NeuroJSON project (https://neurojson.org and https://neurojson.io).

The goal of the NeuroJSON project is to develop scalable, searchable, and
reusable neuroimaging data formats and data sharing platforms. All data
produced from the NeuroJSON project will be using JSON/Binary JData formats as the
underlying serialization standards and the lightweight JData specification as
language-independent data annotation standard.

## File formats

The supported data formats can be found in the below table. All file types
support reading and writing, except those specified below.

| Format | Name |       |  Format                           | Name   |
| ------ | ------ | --- |-----------------------------------| ------ |
| **JSON-compatible files**  | |  | **Binary JSON (same format)** **[1]** | |
| ✅ `.json` | ✅ JSON files |                        | ✅ `.bjd`    | ✅ binary JSON (BJD) files |
| ✅ `.jnii` | ✅ JSON-wrapper for NIfTI data (JNIfTI)|       | ✅ `.bnii`   | ✅ BJD-wrapper for NIfTI data |
| ✅ `.jnirs` | ✅ JSON-wrapper for SNIRF data (JSNIRF)|      | ✅ `.bnirs`  | ✅ BJD-wrapper for SNIRF data |
| ✅ `.jmsh` | ✅ JSON-encoded mesh data (JMesh)  |   | ✅ `.bmsh`   | ✅ BJD-encoded for mesh data  |
| ✅ `.jdt` | ✅ JSON files with JData annotations |  | ✅ `.jdb`    | ✅ BJD files with JData annotations |
| ✅ `.jdat` | ✅ JSON files with JData annotations | | ✅ `.jbat`   | ✅ BJD files with JData annotations |
| ✅ `.jbids` | ✅ JSON digest of a BIDS dataset |    | ✅ `.pmat`   | ✅ BJD encoded .mat files |
| **NIfTI formats**                      | |           | **CSV/TSV formats** | |
| ✅ `.nii` | ✅ uncompressed NIfTI-1/2 files |       | ✅ `.csv`    | ✅ CSV files |
| ✅ `.nii.gz` | ✅ compressed NIfTI files |          | ✅ `.csv.gz` | ✅ compressed CSV files |
| ✅ `.img/.hdr` | ✅ Analyze 7.5 two-part files |    | ✅ `.tsv`    | ✅ TSV files |
| ✅ `.img.gz` | ✅ compressed Analyze files |        | ✅ `.tsv.gz` | ✅ compressed TSV files |
| **HDF5 formats** **[2]**              | |           | **Other formats (read-only)** | |
| ✅ `.h5` | ✅ HDF5 files |                          | ✅ `.mat`    | ✅ MATLAB .mat files **[3]** |
| ✅ `.hdf5` | ✅ HDF5 files |                        | ✅ `.bval`   | ✅ EEG .bval files |
| ✅ `.snirf` | ✅ HDF5-based SNIRF data |            | ✅ `.bvec`   | ✅ EEG .bvec files |
| ✅ `.nwb` | ✅ HDF5-based NWB files |               | ✅ `.msgpack`| ✅ Binary JSON MessagePack format **[4]** |

- [1] requires `bjdata` Python module when needed, `pip install bjdata`
- [2] requires `h5py` Python module when needed, `pip install h5py`
- [3] requires `scipy` Python module when needed, `pip install scipy`
- [4] requires `msgpack` Python module when needed, `pip install msgpack`

## Submodules

The `jdata` module further partition the functions into smaller submodules, including
- **jdata.jfile** provides `loadjd`, `savejd`, `load`, `save`, `loadt`, `savet`, `loadb`, `saveb`, `loadts`, `loadbs`, `jsoncache`, `jdlink`, ...
- **jdata.jdata** provides `encode`, `decode`, `jdataencode`, `jdatadecode`, `{zlib,gzip,lzma,lz4,base64}encode`, `{zlib,gzip,lzma,lz4,base64}decode`
- **jdata.jpath** provides `jsonpath`
- **jdata.jnifti** provides `load{jnifti,nifti}`, `save{jnifti,nifti,jnii,bnii}`, `nii2jnii`, `jnii2nii`, `nifticreate`, `jnifticreate`, `niiformat`, `niicodemap`
- **jdata.neurojson** provides `neuroj`, `neurojgui`
- **jdata.h5** provides `loadh5`, `saveh5`, `regrouph5`, `aos2soa`, `soa2aos`, `jsnirfcreate`, `snirfcreate`, `snirfdecode`

All these functions can be found in the MATLAB/GNU Octave equivalent, JSONLab toolbox. Each function can be individually imported
```
# individually imported
from jdata.jfile import loadjd
data=loadjd(...)

# import everything
from jdata import *
data=loadjd(...)

# import under jdata namespace
import jdata as jd
data=jd.loadjd(...)
```

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
* (optional) **h5py**: PIP: run `pip install h5py`, only needed when reading/writing .h5 and .snirf files
* (optional) **scipy**: PIP: run `pip install scipy`, only needed when loading MATLAB .mat files
* (optional) **msgpack**: PIP: run `pip install msgpack`, only needed when loading MessagePack .msgpack files
* (optional) **blosc2**: PIP: run `pip install blosc2`, only needed when encoding/decoding blosc2-compressed data
* (optional) **backports.lzma**: PIP: run `sudo apt-get install liblzma-dev` and `pip install backports.lzma` (needed for Python 2.7), only needed when encoding/decoding lzma-compressed data
* (optional) **python3-tk**: run `sudo apt-get install python3-tk` to install the Tk support on a Linux in order to run `neurojgui` function

Replacing `pip` by `pip3` if you are using Python 3.x. If either `pip` or `pip3` 
does not exist on your system, please run
```
sudo apt-get install python3-pip
```
Please note that in some OS releases (such as Ubuntu 20.04), python2.x and python-pip 
are no longer supported.

## How to build

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

Instead of installing the module, you can also import the jdata module directly from 
your local copy by cd the root folder of the unzipped pyjdata package, and run
```
import jdata as jd
```


## How to use

The `jdata` module provides a unified data parsing and saving interface: `jd.loadjd()` and `jd.savejd()`.
These two functions supports all file format described in the above "File formats" section.
The `jd.loadjd()` function also supports loading online data via URLs.

```
import jdata as jd
nii = jd.loadjd('/path/to/img.nii.gz')
snirf = jd.loadjd('/path/to/mydata.snirf')
nii2 = jd.loadjd('https://example.com/data/vol.nii.gz')
jsondata = jd.loadjd('https://example.com/rest/api/')
matlabdata = jd.loadjd('matlabdata.mat')
jd.savejd(matlabdata, 'newdata.mat')
jd.savejd(matlabdata, 'newdata.jdb', compression='zlib')

jd.savejd(nii2, 'newdata.jnii', compression='lzma')
jd.savejd(nii, 'newdata.bnii', compression='gzip')
jd.savejd(nii, 'newdata.nii.gz')
```

The `jdata` module also serves as the front-end for the free data resources hosted at
NeuroJSON.io. The NeuroJSON client (`neuroj()`) can be started in the GUI mode using

```
import jdata as jd
jd.neuroj('gui')
```

the above command will pop up a window displaying the databases, datasets and data
records for the over 1500 datasets currently hosted on NeuroJSON.io.

The `neuroj` client also supports command-line mode, using the below format

```
import jdata as jd
help(jd.neuroj)                            # print help info for jd.neuroj()
jd.neuroj('list')                          # list all databases on NeuroJSON.io
[db['id'] for db in jd.neuroj('list')['database']]  # list all database IDs
jd.neuroj('list', 'openneuro')             # list all datasets under the `openneuro` database
jd.neuroj('list', 'openneuro', limit=5, skip=5)  # list the 6th to 10th datasets under the `openneuro` database
jd.neuroj('list', 'openneuro', 'ds000001') # list all versions for the `openneuro/ds00001` dataset
jd.neuroj('get', 'openneuro', 'ds000001')  # download and parse the `openneuro/ds00001` dataset as a Python object
jd.neuroj('info', 'openneuro', 'ds000001') # lightweight header information of the `openneuro/ds00001` dataset
jd.neuroj('find', '/abide/')               # find both abide-1 and abide-2 databases using filters
jd.neuroj('find', 'openneuro', '/00[234]$/') # use regular experssion to filter all openneuro datasets
jd.neuroj('find', 'mcx', {'selector': ..., 'find': ...}) # use CouchDB _find API to search data
jd.neuroj('find', 'mcx', {'selector': ..., 'find': ...}) # use CouchDB _find API to search data
jd.neuroj('info', db='mcx', ds='colin27')  # use named inputs
jd.neuroj('get', db='mcx', ds='colin27', file='att1')  # download the attachment `att1` for the `mcx/colin27` dataset
jd.neuroj('put', 'sandbox1d', 'test', '{"obj":1}')  # update `sandbox1d/test` dataset with a new JSON string (need admin account)
jd.neuroj('delete', 'sandbox1d', 'test')   # delete `sandbox1d/test` dataset (need admin account)
```


## Advanced interfaces

The `jdata` module is easy to use. You can use the `encode()/decode()` functions to
encode Python data into JData annotation format, or decode JData structures into
native Python data, for example

```
import jdata as jd
import numpy as np

a={'str':'test','num':1.2,'list':[1.1,[2.1]],'nan':float('nan'),'np':np.arange(1,5,dtype=np.uint8)}
jd.encode(a)
jd.decode(jd.encode(a))
d1=jd.encode(a, compression='zlib',base64=True})
d1
jd.decode(d1,base64=True)
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
python3 -m jdata /path/to/file.json           # convert to /path/to/text/json/file.jdb
python3 -m jdata /path/to/file.jdb            # convert to /path/to/text/json/file.json
python3 -m jdata /path/to/file.jdb -t 2       # convert to /path/to/text/json/file.json with indentation of 2 spaces
python3 -m jdata file1 file2 ...              # batch convert multiple files
python3 -m jdata file1 -f                     # force overwrite output files if exist (`-f`/`--force`)
python3 -m jdata file1 -O /output/dir         # save output files to /output/dir (`-O`/`--outdir`)
python3 -m jdata file1.json -s .bnii          # force output suffix/file type (`-s`/`--suffix`)
python3 -m jdata file1.json -c zlib           # set compression method (`-c`/`--compression`)
python3 -m jdata -h                           # show help info (`-h`/`--help`)
```

## How to contribute

`jdata` uses an open-source license - the Apache 2.0 license. This license is a "permissive" license
and can be used in commercial products without needing to release the source code.

To contribute `jdata` source code, you can modify the Python units inside the `jdata/` folder. Please
minimize the dependencies to external 3rd party packages. Please use Python's built-in packages whenever
pissible.

All jdata source codes have been formatted using `black`. To reformat all units, please type
```
make pretty
```
inside the top-folder of the source repository

For every newly added function, please add a unittest unit or test inside the files under `test/`, and run
```
make test
```
to make sure the modified code can pass all tests.

To build a local installer, please install the `build` python module, and run
```
make build
```
The output wheel can be found inside the `dist/` folder.

## Test

To see additional data type support, please run the built-in test using below command

```
python3 -m unittest discover -v test
```
or one can run individual set of unittests by calling
```
python3 -m unittest -v test.testnifti
python3 -m unittest -v test.testsnirf
```
