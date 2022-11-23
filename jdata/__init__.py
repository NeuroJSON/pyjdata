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

from .jfile import load, save, show, loadt, savet, loadb, saveb, jext
from .jdata import encode, decode, jdtype, jsonfilter

__version__ = "0.5.2"
__all__ = [
    "load",
    "save",
    "show",
    "loadt",
    "savet",
    "loadb",
    "saveb",
    "encode",
    "decode",
    "jdtype",
    "jsonfilter",
    "jext",
]
__license__ = """Apache license 2.0, Copyright (c) 2019-2022 Qianqian Fang"""


if __name__ == "__main__":
    import cmd

    cmd.main()
