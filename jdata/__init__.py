"""pyjdata - encode and decode Python data structrues using portable JData format

This module provides an encoder and a decoder to convert a python/numpy native 
data structure into a JData-compatible structure, or decode JData constructs to 
restore python native data.

    jdata = encode(pydata)
    newpydata = decode(jdata)

    save(jdata,filename)
    jdata=load(filename)

The function ``encode`` converts the below python/numpy/pandas data types
into JData compatible dict-based objects

* python dict objects ==> unchanged (JData/JSON objects)
* python string objects ==> unchanged (JData/JSON strings)
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

from .jfile import load, save, loadt, savet, loadb, saveb
from .jdata import encode, decode, jdtype, jsonfilter

__version__ = '0.2'
__all__ = ['load','save','loadt', 'savet', 'loadb', 'saveb','encode', 'decode', 'jdtype','jsonfilter']
__license__ = """Apache license 2.0, Copyright (c) 2019 Qianqian Fang"""


if __name__ == '__main__':
    import cmd
    cmd.main()
