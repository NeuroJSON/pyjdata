"""@package docstring
Encoding and decoding python native data structures as
portable JData-spec annotated dict structure

Copyright (c) 2019-2022 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["encode", "decode", "jdtype", "jsonfilter"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import copy
import zlib
import base64

##====================================================================================
## global variables
##====================================================================================

""" @brief Mapping Numpy data types to JData data types
complex-valued data are reflected in the doubled data size
"""

jdtype = {
    "float32": "single",
    "float64": "double",
    "float_": "double",
    "bool": "uint8",
    "byte": "int8",
    "short": "int16",
    "ubyte": "uint8",
    "ushort": "uint16",
    "int_": "int32",
    "uint": "uint32",
    "complex_": "double",
    "complex128": "double",
    "complex64": "single",
    "longlong": "int64",
    "ulonglong": "uint64",
    "csingle": "single",
    "cdouble": "double",
}

_zipper = (
    "zlib",
    "gzip",
    "lzma",
    "lz4",
    "blosc2blosclz",
    "blosc2lz4",
    "blosc2lz4hc",
    "blosc2zlib",
    "blosc2zstd",
    "base64",
)

_allownumpy = ("_ArraySize_", "_ArrayData_", "_ArrayZipSize_", "_ArrayZipData_")

##====================================================================================
## Python to JData encoding function
##====================================================================================


def encode(d, opt={}):
    """@brief Encoding a Python data structure to portable JData-annotated dict constructs

    This function converts complex data types (usually not JSON-serializable) into
    portable JData-annotated dict/list constructs that can be easily exported as JSON/JData
    files

    @param[in,out] d: an arbitrary Python data
    @param[in] opt: options, can contain a dict with the following keys
         'compression': choose one of ['zlib','lzma','gzip','lz4','blosc2blosclz','blosc2lz4',
                        'blosc2lz4hc','blosc2zlib','blosc2zstd'] for compression codec, default is None
         'nthread': number of compression thread of the codec is of the blosc2 class, default is 1
    """

    opt.setdefault("inplace", False)

    if "compression" in opt:
        if opt["compression"] == "lzma":
            try:
                try:
                    import lzma
                except ImportError:
                    from backports import lzma
            except Exception:
                raise Exception(
                    "JData",
                    'you must install "lzma" module to compress with this format',
                )
        elif opt["compression"] == "lz4":
            try:
                import lz4.frame
            except ImportError:
                raise Exception(
                    "JData",
                    'you must install "lz4" module to compress with this format',
                )
        elif opt["compression"].startswith("blosc2"):
            try:
                import blosc2
            except ImportError:
                raise Exception(
                    "JData",
                    'you must install "blosc2" module to compress with this format',
                )

    if isinstance(d, float):
        if np.isnan(d):
            return "_NaN_"
        elif np.isinf(d):
            return "_Inf_" if (d > 0) else "-_Inf_"
        return d
    elif isinstance(d, list) or isinstance(d, tuple) or isinstance(d, set) or isinstance(d, frozenset):
        return encodelist(d, opt)
    elif isinstance(d, dict):
        return encodedict(d, opt)
    elif isinstance(d, complex):
        newobj = {
            "_ArrayType_": "double",
            "_ArraySize_": 1,
            "_ArrayIsComplex_": True,
            "_ArrayData_": [d.real, d.imag],
        }
        return newobj
    elif isinstance(d, np.ndarray) or np.iscomplex(d):
        newobj = {}
        newobj["_ArrayType_"] = jdtype[str(d.dtype)] if (str(d.dtype) in jdtype) else str(d.dtype)
        if np.isscalar(d):
            newobj["_ArraySize_"] = 1
        else:
            newobj["_ArraySize_"] = list(d.shape)
        if d.dtype == np.complex64 or d.dtype == np.complex128 or d.dtype == np.csingle or d.dtype == np.cdouble:
            newobj["_ArrayIsComplex_"] = True
            newobj["_ArrayData_"] = np.stack((d.ravel().real, d.ravel().imag))
        else:
            newobj["_ArrayData_"] = d.ravel()

        if "compression" in opt:
            if opt["compression"] not in _zipper:
                raise Exception(
                    "JData",
                    "compression method {} is not supported".format(opt["compression"]),
                )
            newobj["_ArrayZipType_"] = opt["compression"]
            newobj["_ArrayZipSize_"] = [1 + int("_ArrayIsComplex_" in newobj), d.size]
            newobj["_ArrayZipData_"] = newobj["_ArrayData_"].data
            if opt["compression"] == "zlib":
                newobj["_ArrayZipData_"] = zlib.compress(newobj["_ArrayZipData_"])
            elif opt["compression"] == "gzip":
                gzipper = zlib.compressobj(wbits=(zlib.MAX_WBITS | 16))
                newobj["_ArrayZipData_"] = gzipper.compress(newobj["_ArrayZipData_"])
            elif opt["compression"] == "lzma":
                try:
                    newobj["_ArrayZipData_"] = lzma.compress(newobj["_ArrayZipData_"], lzma.FORMAT_ALONE)
                except Exception:
                    print('you must install "lzma" module to compress with this format, ignoring')
                    pass
            elif opt["compression"] == "lz4":
                try:
                    newobj["_ArrayZipData_"] = lz4.frame.compress(newobj["_ArrayZipData_"].tobytes())
                except ImportError:
                    print('you must install "lz4" module to compress with this format, ignoring')
                    pass
            elif opt["compression"].startswith("blosc2"):
                try:
                    BLOSC2CODEC = {
                        "blosc2blosclz": blosc2.Codec.BLOSCLZ,
                        "blosc2lz4": blosc2.Codec.LZ4,
                        "blosc2lz4hc": blosc2.Codec.LZ4HC,
                        "blosc2zlib": blosc2.Codec.ZLIB,
                        "blosc2zstd": blosc2.Codec.ZSTD,
                    }
                    blosc2nthread = 1
                    if "nthread" in opt:
                        blosc2nthread = opt["nthread"]
                    newobj["_ArrayZipData_"] = blosc2.compress2(
                        newobj["_ArrayZipData_"],
                        compcode=BLOSC2CODEC[opt["compression"]],
                        typesize=d.dtype.itemsize,
                        nthreads=blosc2nthread,
                    )
                except ImportError:
                    print('you must install "blosc2" module to compress with this format, ignoring')
                    pass
            if (("base64" in opt) and (opt["base64"])) or opt["compression"] == "base64":
                newobj["_ArrayZipData_"] = base64.b64encode(newobj["_ArrayZipData_"])
            newobj.pop("_ArrayData_")
        return newobj
    else:
        return copy.deepcopy(d) if opt["inplace"] else d


##====================================================================================
## JData to Python decoding function
##====================================================================================


def decode(d, opt={}):
    """@brief Decoding a JData-annotated dict construct into native Python data

    This function converts portable JData-annotated dict/list constructs back to native Python
    data structures

    @param[in,out] d: an arbitrary Python data, any JData-encoded components will be decoded
    @param[in] opt: options, can contain a dict with the following keys
         'nthread': number of decompression thread of the codec is of the blosc2 class, default is 1
    """

    opt.setdefault("inplace", False)

    if (isinstance(d, str) or type(d) == "unicode") and len(d) <= 6 and len(d) > 4 and d[-1] == "_":
        if d == "_NaN_":
            return float("nan")
        elif d == "_Inf_":
            return float("inf")
        elif d == "-_Inf_":
            return float("-inf")
        return d
    elif isinstance(d, list) or isinstance(d, tuple) or isinstance(d, set) or isinstance(d, frozenset):
        return decodelist(d, opt)
    elif isinstance(d, dict):
        if "_ArrayType_" in d:
            if isinstance(d["_ArraySize_"], str):
                d["_ArraySize_"] = np.frombuffer(bytearray(d["_ArraySize_"]))
            if "_ArrayZipData_" in d:
                newobj = d["_ArrayZipData_"]
                if (("base64" in opt) and (opt["base64"])) or ("_ArrayZipType_" in d and d["_ArrayZipType_"] == "base64"):
                    newobj = base64.b64decode(newobj)
                if "_ArrayZipType_" in d and d["_ArrayZipType_"] not in _zipper:
                    raise Exception(
                        "JData",
                        "compression method {} is not supported".format(d["_ArrayZipType_"]),
                    )
                if d["_ArrayZipType_"] == "zlib":
                    newobj = zlib.decompress(bytes(newobj))
                elif d["_ArrayZipType_"] == "gzip":
                    newobj = zlib.decompress(bytes(newobj), zlib.MAX_WBITS | 32)
                elif d["_ArrayZipType_"] == "lzma":
                    try:
                        import lzma
                    except ImportError:
                        from backports import lzma
                    buf = bytearray(newobj)  # set length to -1 (unknown) if EOF appears
                    buf[5:13] = b"\xff\xff\xff\xff\xff\xff\xff\xff"
                    newobj = lzma.decompress(buf, lzma.FORMAT_ALONE)
                elif d["_ArrayZipType_"] == "lz4":
                    try:
                        import lz4.frame

                        newobj = lz4.frame.decompress(bytes(newobj))
                    except Exception:
                        print('Warning: you must install "lz4" module to decompress a data record in this file, ignoring')
                        return copy.deepcopy(d) if opt["inplace"] else d
                elif d["_ArrayZipType_"].startswith("blosc2"):
                    try:
                        import blosc2

                        blosc2nthread = 1
                        if "nthread" in opt:
                            blosc2nthread = opt["nthread"]
                        newobj = blosc2.decompress2(bytes(newobj), as_bytearray=False, nthreads=blosc2nthread)
                    except Exception:
                        print('Warning: you must install "blosc2" module to decompress a data record in this file, ignoring')
                        return copy.deepcopy(d) if opt["inplace"] else d
                newobj = np.frombuffer(bytearray(newobj), dtype=np.dtype(d["_ArrayType_"])).reshape(d["_ArrayZipSize_"])
                if "_ArrayIsComplex_" in d and newobj.shape[0] == 2:
                    newobj = newobj[0] + 1j * newobj[1]
                if "_ArrayOrder_" in d and (
                    d["_ArrayOrder_"].lower() == "c" or d["_ArrayOrder_"].lower() == "col" or d["_ArrayOrder_"].lower() == "column"
                ):
                    newobj = newobj.reshape(d["_ArraySize_"], order="F")
                else:
                    newobj = newobj.reshape(d["_ArraySize_"])
                if not hasattr(d["_ArraySize_"], "__iter__") and d["_ArraySize_"] == 1:
                    newobj = newobj.item()
                return newobj
            elif "_ArrayData_" in d:
                if isinstance(d["_ArrayData_"], str):
                    newobj = np.frombuffer(d["_ArrayData_"], dtype=np.dtype(d["_ArrayType_"]))
                else:
                    newobj = np.asarray(d["_ArrayData_"], dtype=np.dtype(d["_ArrayType_"]))
                if "_ArrayZipSize_" in d and newobj.shape[0] == 1:
                    if isinstance(d["_ArrayZipSize_"], str):
                        d["_ArrayZipSize_"] = np.frombuffer(bytearray(d["_ArrayZipSize_"]))
                    newobj = newobj.reshape(d["_ArrayZipSize_"])
                if "_ArrayIsComplex_" in d and newobj.shape[0] == 2:
                    newobj = newobj[0] + 1j * newobj[1]
                if "_ArrayOrder_" in d and (
                    d["_ArrayOrder_"].lower() == "c" or d["_ArrayOrder_"].lower() == "col" or d["_ArrayOrder_"].lower() == "column"
                ):
                    newobj = newobj.reshape(d["_ArraySize_"], order="F")
                else:
                    newobj = newobj.reshape(d["_ArraySize_"])
                if not hasattr(d["_ArraySize_"], "__iter__") and d["_ArraySize_"] == 1:
                    newobj = newobj.item()
                return newobj
            else:
                raise Exception(
                    "JData",
                    "one and only one of _ArrayData_ or _ArrayZipData_ is required",
                )
        return decodedict(d, opt)
    else:
        return copy.deepcopy(d) if opt["inplace"] else d


##====================================================================================
## helper functions
##====================================================================================


def jsonfilter(obj):
    if type(obj) == "long":
        return str(obj)
    elif type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    elif isinstance(obj, float):
        if np.isnan(obj):
            return "_NaN_"
        elif np.isinf(obj):
            return "_Inf_" if (obj > 0) else "-_Inf_"


# -------------------------------------------------------------------------------------


def encodedict(d0, opt={}):
    d = dict(d0)
    for k, v in d0.items():
        if isinstance(v, np.ndarray) and isinstance(k, str) and (k in _allownumpy):
            continue
        newkey = encode(k, opt)
        d[newkey] = encode(v, opt)
        if k != newkey:
            d.pop(k)
    return d


# -------------------------------------------------------------------------------------


def encodelist(d0, opt={}):
    d = copy.deepcopy(d0) if opt["inplace"] else d0
    for i, s in enumerate(d):
        d[i] = encode(s, opt)
    return d


# -------------------------------------------------------------------------------------


def decodedict(d0, opt={}):
    d = dict(d0)
    for k, v in d.items():
        newkey = encode(k, opt)
        d[newkey] = decode(v, opt)
        if k != newkey:
            d.pop(k)
    return d


# -------------------------------------------------------------------------------------


def decodelist(d0, opt={}):
    d = copy.deepcopy(d0) if opt["inplace"] else d0
    for i, s in enumerate(d):
        d[i] = decode(s, opt)
    return d


# -------------------------------------------------------------------------------------
