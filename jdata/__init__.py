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

from . import jdata
from . import jfile
from . import jdictionary
from . import csvtsv
from . import h5
from . import jnifti
from . import jgifti
from . import jpath
from . import jschema
from . import njprep
from . import neurojson

# Re-export all public functions from submodules
from .jdata import *
from .jfile import *
from .jdictionary import *
from .csvtsv import *
from .h5 import *
from .jnifti import *
from .jgifti import *
from .jpath import *
from .jschema import *
from .njprep import *
from .neurojson import *

__version__ = "0.9.3"
__all__ = (
    jdata.__all__
    + jfile.__all__
    + jdictionary.__all__
    + csvtsv.__all__
    + h5.__all__
    + jnifti.__all__
    + jgifti.__all__
    + jpath.__all__
    + jschema.__all__
    + njprep.__all__
    + neurojson.__all__
    + [
        "jdata",
        "jfile",
        "jdictionary",
        "csvtsv",
        "h5",
        "jnifti",
        "jgifti",
        "jpath",
        "jschema",
        "neurojson",
        "njprep",
    ]
)

__license__ = """Apache license 2.0, Copyright (c) 2019-2026 Qianqian Fang"""
