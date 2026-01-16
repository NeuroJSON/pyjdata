"""pyjdata - encode and decode Python data structrues using portable JData formats

This module provides an encoder and a decoder to convert a python/numpy native
data structure into a JData-compatible structure, or decode JData constructs to
restore python native data.

    import jdata as jd

    jdata = jd.encode(pydata)
    newpydata = jd.decode(jdata)

    jd.save(jdata,filename)
    jdata=jd.load(filename)

The function ``encode`` converts the below python/numpy/pandas data types
into JData compatible dict-based objects

* python dict objects ==> unchanged (JData/JSON objects)
* python string objects ==> unchanged (JData/JSON strings)
* python unicode objects ==> unchanged (JData/JSON strings)
* python numeric objects ==> unchanged (JData/JSON values)
* python list objects ==> unchanged (JData/JSON arrays)
* python tuple objects ==> unchanged (JData/JSON arrays)
* python range objects ==> unchanged (JData/JSON arrays)
* python bytes/bytearray objects ==> unchanged (JData/JSON bytestream)
* python set objects ==> unchanged (JData/JSON arrays)
* python complex numbers and complex arrays ==> JData complex array objects
* numpy.ndarray objects ==> JData array objects
* python tables ==> JData table objects
* pandas.DataFrame objects ==> JData table objects

The JData-encoded data object can then be decoded using ``decode``
to restore the original data types

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

from .jfile import (
    loadjson,
    savejson,
    loadbj,
    savebj,
    loadjd,
    savejd,
    load,
    save,
    loadurl,
    show,
    dumpb,
    loadt,
    savet,
    loadts,
    loadbs,
    loadb,
    saveb,
    jsoncache,
    jdlink,
    jext,
    loadjsnirf,
    loadsnirf,
    savejsnirf,
    savesnirf,
    loadmsgpack,
    savemsgpack,
    loadubjson,
    saveubjson,
)
from .jdata import (
    jdataencode,
    jdatadecode,
    encode,
    decode,
    jdtype,
    jsonfilter,
    zlibencode,
    zlibdecode,
    gzipencode,
    gzipdecode,
    lzmaencode,
    lzmadecode,
    lz4encode,
    lz4decode,
    base64encode,
    base64decode,
)
from .jpath import jsonpath
from .jnifti import (
    nii2jnii,
    jnii2nii,
    loadnifti,
    loadjnifti,
    savenifti,
    savejnifti,
    nifticreate,
    jnifticreate,
    memmapstream,
    niiheader2jnii,
    niicodemap,
    niiformat,
    savejnii,
    savebnii,
)

from .h5 import (
    loadh5,
    saveh5,
    regrouph5,
    aos2soa,
    soa2aos,
    jsnirfcreate,
    snirfcreate,
    snirfdecode,
)

from .csv import (
    load_csv_tsv,
    loadcsv,
    loadtsv,
    save_csv_tsv,
)

from .jdict import jdict
from .jschema import jsonschema
from .neurojson import neuroj, neurojgui

__version__ = "0.9.1"
__all__ = [
    "loadjson",
    "savejson",
    "loadbj",
    "savebj",
    "loadjd",
    "savejd",
    "jdataencode",
    "jdatadecode",
    "load",
    "save",
    "loadurl",
    "show",
    "dumpb",
    "loadt",
    "savet",
    "loadts",
    "loadbs",
    "loadb",
    "saveb",
    "encode",
    "decode",
    "jsoncache",
    "jdlink",
    "jdtype",
    "jsonfilter",
    "jext",
    "jsonpath",
    "nii2jnii",
    "jnii2nii",
    "loadnifti",
    "loadjnifti",
    "savenifti",
    "savejnifti",
    "nifticreate",
    "jnifticreate",
    "memmapstream",
    "niiheader2jnii",
    "niicodemap",
    "niiformat",
    "savebnii",
    "savejnii",
    "loadh5",
    "saveh5",
    "regrouph5",
    "aos2soa",
    "soa2aos",
    "jsnirfcreate",
    "snirfcreate",
    "loadjsnirf",
    "loadsnirf",
    "savejsnirf",
    "savesnirf",
    "snirfdecode",
    "zlibencode",
    "zlibdecode",
    "gzipencode",
    "gzipdecode",
    "lzmaencode",
    "lzmadecode",
    "lz4encode",
    "lz4decode",
    "base64encode",
    "base64decode",
    "neuroj",
    "neurojgui",
    "load_csv_tsv",
    "loadcsv",
    "loadtsv",
    "save_csv_tsv",
    "loadmsgpack",
    "savemsgpack",
    "loadubjson",
    "saveubjson",
    "jdict",
    "jsonschema",
]

__license__ = """Apache license 2.0, Copyright (c) 2019-2026 Qianqian Fang"""
