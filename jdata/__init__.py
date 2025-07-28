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
)
from .h5 import loadh5, saveh5

__version__ = "0.7.1"
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
    "loadh5",
    "saveh5",
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
]
__license__ = """Apache license 2.0, Copyright (c) 2019-2024 Qianqian Fang"""


if __name__ == "__main__":
    import cmd

    cmd.main()
