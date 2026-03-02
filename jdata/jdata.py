"""@package docstring
Encoding and decoding python native data structures as
portable JData-spec annotated dict structure

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "encode",
    "decode",
    "jdataencode",
    "jdatadecode",
    "jdtype",
    "jsonfilter",
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

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import copy
import zlib
import base64
import os
import re
from .jpath import jsonpath
import lzma

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

_allownumpy = (
    "_ArraySize_",
    "_ArrayData_",
    "_ArrayZipSize_",
    "_ArrayZipData_",
    "_ArrayIsSparse_",
    "_ArrayIsComplex_",
)

##====================================================================================
## Python to JData encoding function
##====================================================================================


def _compress_data(rawbytes, opt):
    """Compress raw bytes using the codec specified in opt['compression']."""
    codec = opt["compression"]
    if codec == "zlib":
        return zlib.compress(rawbytes)
    elif codec == "gzip":
        gzipper = zlib.compressobj(wbits=(zlib.MAX_WBITS | 16))
        result = gzipper.compress(rawbytes)
        result += gzipper.flush()
        return result
    elif codec == "lzma":
        return lzma.compress(rawbytes, lzma.FORMAT_ALONE)
    elif codec == "lz4":
        import lz4.frame

        return lz4.frame.compress(rawbytes)
    elif codec.startswith("blosc2"):
        import blosc2

        BLOSC2CODEC = {
            "blosc2blosclz": blosc2.Codec.BLOSCLZ,
            "blosc2lz4": blosc2.Codec.LZ4,
            "blosc2lz4hc": blosc2.Codec.LZ4HC,
            "blosc2zlib": blosc2.Codec.ZLIB,
            "blosc2zstd": blosc2.Codec.ZSTD,
        }
        nthread = opt.get("nthread", 1)
        return blosc2.compress2(rawbytes, codec=BLOSC2CODEC[codec], nthreads=nthread)
    elif codec == "base64":
        return rawbytes
    return rawbytes


def _issparse(d):
    """Check if d is a scipy sparse matrix without importing scipy at module level."""
    try:
        import scipy.sparse

        return scipy.sparse.issparse(d)
    except ImportError:
        return False


def encode(d, opt=None, **kwargs):
    """
    Encode a Python data structure to portable JData-annotated dict constructs.

    Converts complex data types (numpy arrays, complex numbers, special floats)
    into JData-annotated dict/list constructs that can be serialized as JSON or
    binary JSON. Scalar types (int, float, str, bool, None) pass through unchanged.

    Args:
        d: An arbitrary Python data structure to encode. Supported types include
            float, int, str, bool, None, list, tuple, set, frozenset, dict,
            complex, numpy.ndarray, and nested combinations thereof.
        opt (dict, optional): Legacy options dict. If provided, its contents are
            merged into kwargs. Prefer passing options as keyword arguments directly.
        **kwargs: Encoding options including:
            compression (str): Compression codec for numpy arrays. One of 'zlib',
                'gzip', 'lzma', 'lz4', 'base64', or blosc2 variants. Default 'zlib'.
            compressarraysize (int): Minimum array size (in elements) to trigger
                compression. Default 200.
            base64 (bool): If True, base64-encode compressed data for text JSON.
            inplace (bool): If True, make deep copies to avoid mutating input.

    Returns:
        The JData-annotated version of the input. Numpy arrays become dicts with
        keys like '_ArrayType_', '_ArraySize_', '_ArrayData_' (or '_ArrayZipData_'
        if compressed). Special floats become '_NaN_', '_Inf_', '-_Inf_'. Other
        types pass through or are recursively encoded.

    Examples:
        >>> import numpy as np
        >>> encode(float('nan'))
        '_NaN_'
        >>> encode(np.array([1, 2, 3], dtype=np.uint8))
        {'_ArrayType_': 'uint8', '_ArraySize_': [3], '_ArrayData_': ...}
        >>> encode({'a': [1, 2], 'b': np.zeros(3)})
        {'a': [1, 2], 'b': {'_ArrayType_': 'double', ...}}
    """
    if opt is None:
        opt = {}
    kwargs.setdefault("compression", "zlib")
    kwargs.setdefault("compressarraysize", 300)
    opt.setdefault("inplace", False)
    opt.update(kwargs)

    if "compression" in opt:
        if opt["compression"] == "lzma":
            try:
                try:
                    import lzma
                except ImportError:
                    from backports import lzma
            except ImportError:
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
    elif isinstance(d, list) or isinstance(d, set):
        return encodelist(d, **opt)
    elif isinstance(d, tuple) or isinstance(d, frozenset):
        return encodelist(list(d), **opt)
    elif isinstance(d, dict):
        return encodedict(d, **opt)
    elif isinstance(d, complex):
        newobj = {
            "_ArrayType_": "double",
            "_ArraySize_": 1,
            "_ArrayIsComplex_": True,
            "_ArrayData_": [d.real, d.imag],
        }
        return newobj
    elif _issparse(d):
        import scipy.sparse

        coo = d.tocoo()
        newobj = {}
        val_dtype = coo.data.dtype if len(coo.data) > 0 else np.float64
        if np.issubdtype(val_dtype, np.complexfloating):
            real_dtype = val_dtype.type(0).real.dtype
        else:
            real_dtype = val_dtype
        newobj["_ArrayType_"] = jdtype.get(str(real_dtype), str(real_dtype))
        newobj["_ArraySize_"] = list(d.shape)
        newobj["_ArrayIsSparse_"] = True
        if np.issubdtype(val_dtype, np.complexfloating):
            newobj["_ArrayIsComplex_"] = True
            newobj["_ArrayData_"] = [
                (coo.row + 1).astype(np.float64).tolist(),
                (coo.col + 1).astype(np.float64).tolist(),
                coo.data.real.astype(np.float64).tolist(),
                coo.data.imag.astype(np.float64).tolist(),
            ]
        else:
            newobj["_ArrayData_"] = [
                (coo.row + 1).astype(np.float64).tolist(),
                (coo.col + 1).astype(np.float64).tolist(),
                coo.data.astype(np.float64).tolist(),
            ]
        if "compression" in opt and opt["compression"] in _zipper:
            arraydata = np.array(newobj["_ArrayData_"])
            nrows = arraydata.shape[0]
            nnz = arraydata.shape[1] if arraydata.ndim > 1 else 0
            if nnz >= opt.get("compressarraysize", 300):
                rawbytes = arraydata.astype(np.float64).tobytes()
                newobj["_ArrayZipType_"] = opt["compression"]
                newobj["_ArrayZipSize_"] = [nrows, nnz]
                newobj["_ArrayZipData_"] = _compress_data(rawbytes, opt)
                if (("base64" in opt) and (opt["base64"])) or opt["compression"] == "base64":
                    newobj["_ArrayZipData_"] = base64.b64encode(newobj["_ArrayZipData_"])
                newobj.pop("_ArrayData_")
        return newobj
    elif isinstance(d, np.ndarray) or np.iscomplex(d):
        newobj = {}
        newobj["_ArrayType_"] = jdtype[str(d.dtype)] if (str(d.dtype) in jdtype) else str(d.dtype)
        if np.isscalar(d):
            newobj["_ArraySize_"] = 1
        else:
            newobj["_ArraySize_"] = list(d.shape)
        if (
            d.dtype == np.complex64
            or d.dtype == np.complex128
            or d.dtype == np.csingle
            or d.dtype == np.cdouble
        ):
            newobj["_ArrayIsComplex_"] = True
            newobj["_ArrayData_"] = np.stack((d.ravel().real, d.ravel().imag))
        else:
            newobj["_ArrayData_"] = d.ravel()

        if "compression" in opt and d.size >= opt.get("compressarraysize", 300):
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
                newobj["_ArrayZipData_"] += gzipper.flush()
            elif opt["compression"] == "lzma":
                try:
                    newobj["_ArrayZipData_"] = lzma.compress(
                        newobj["_ArrayZipData_"], lzma.FORMAT_ALONE
                    )
                except Exception:
                    print('you must install "lzma" module to compress with this format, ignoring')
                    pass
            elif opt["compression"] == "lz4":
                try:
                    newobj["_ArrayZipData_"] = lz4.frame.compress(
                        newobj["_ArrayZipData_"].tobytes()
                    )
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
                        codec=BLOSC2CODEC[opt["compression"]],
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


def decode(d, opt=None, **kwargs):
    """
    Decode JData-annotated dict constructs back into native Python data.

    Reverses the encoding performed by encode(). Recognizes JData annotation keys
    ('_ArrayType_', '_ArraySize_', '_ArrayData_', '_ArrayZipData_', etc.) and
    reconstructs numpy arrays, complex numbers, and special float values.

    Args:
        d: A JData-annotated Python data structure (dict, list, or scalar).
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Decoding options including:
            base64 (bool): If True, expect base64-encoded compressed data.
            inplace (bool): If True, make deep copies to avoid mutating input.
            maxlinklevel (int): Maximum depth for resolving '_DataLink_' references.

    Returns:
        The decoded native Python data structure with numpy arrays, complex numbers,
        and special floats restored.

    Examples:
        >>> decode('_NaN_')
        nan
        >>> decode({'_ArrayType_': 'uint8', '_ArraySize_': [3], '_ArrayData_': [1, 2, 3]})
        array([1, 2, 3], dtype=uint8)
    """

    if opt is None:
        opt = {}
    from .jfile import jdlink

    opt.setdefault("inplace", False)
    opt.setdefault("maxlinklevel", 0)
    opt.update(kwargs)

    if (isinstance(d, str) or type(d) == "unicode") and len(d) <= 6 and len(d) > 4 and d[-1] == "_":
        if d == "_NaN_":
            return float("nan")
        elif d == "_Inf_":
            return float("inf")
        elif d == "-_Inf_":
            return float("-inf")
        return d
    elif isinstance(d, list) or isinstance(d, set):
        return decodelist(d, **opt)
    elif isinstance(d, tuple) or isinstance(d, frozenset):
        return decodelist(list(d), **opt)
    elif isinstance(d, dict):
        if "_ArrayType_" in d:
            # Early intercept for sparse arrays
            if "_ArrayIsSparse_" in d and d["_ArrayIsSparse_"]:
                try:
                    import scipy.sparse
                except ImportError:
                    raise ImportError('To decode sparse JData, install scipy: "pip install scipy"')
                shape = (
                    tuple(d["_ArraySize_"])
                    if isinstance(d["_ArraySize_"], list)
                    else (d["_ArraySize_"],)
                )
                is_complex = "_ArrayIsComplex_" in d and d["_ArrayIsComplex_"]
                if "_ArrayZipData_" in d:
                    # Decompress first
                    newobj = d["_ArrayZipData_"]
                    if isinstance(newobj, str):
                        newobj = newobj.encode("ascii")
                    if ("base64" in opt and opt["base64"]) or (
                        "_ArrayZipType_" in d and d["_ArrayZipType_"] == "base64"
                    ):
                        newobj = base64.b64decode(newobj)
                    if "_ArrayZipType_" in d and d["_ArrayZipType_"] != "base64":
                        if d["_ArrayZipType_"] == "zlib":
                            newobj = zlib.decompress(newobj)
                        elif d["_ArrayZipType_"] == "gzip":
                            newobj = zlib.decompress(newobj, zlib.MAX_WBITS | 16)
                        elif d["_ArrayZipType_"] == "lzma":
                            buf = bytearray(newobj)
                            if len(buf) > 13:
                                buf[5:13] = b"\xff\xff\xff\xff\xff\xff\xff\xff"
                            newobj = lzma.decompress(buf, lzma.FORMAT_ALONE)
                        elif d["_ArrayZipType_"] == "lz4":
                            import lz4.frame

                            newobj = lz4.frame.decompress(bytes(newobj))
                        elif d["_ArrayZipType_"].startswith("blosc2"):
                            import blosc2

                            nthread = opt.get("nthread", 1)
                            newobj = blosc2.decompress2(
                                bytes(newobj), as_bytearray=False, nthreads=nthread
                            )
                    arraydata = np.frombuffer(bytearray(newobj), dtype=np.float64).reshape(
                        d["_ArrayZipSize_"]
                    )
                else:
                    arraydata = np.array(d["_ArrayData_"], dtype=np.float64)
                if arraydata.ndim == 1:
                    nrows = 4 if is_complex else 3
                    nnz = len(arraydata) // nrows
                    arraydata = arraydata.reshape(nrows, nnz)
                rows = arraydata[0].astype(np.intp) - 1
                cols = arraydata[1].astype(np.intp) - 1
                if is_complex:
                    vals = arraydata[2] + 1j * arraydata[3]
                else:
                    vals = arraydata[2]
                return scipy.sparse.csc_matrix((vals, (rows, cols)), shape=shape)

            if isinstance(d["_ArraySize_"], str):
                d["_ArraySize_"] = np.frombuffer(bytearray(d["_ArraySize_"]))
            if "_ArrayZipData_" in d:
                newobj = d["_ArrayZipData_"]
                if (("base64" in opt) and (opt["base64"])) or (
                    "_ArrayZipType_" in d and d["_ArrayZipType_"] == "base64"
                ):
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
                    buf = bytearray(newobj)  # set length to -1 (unknown) if EOF appears
                    buf[5:13] = b"\xff\xff\xff\xff\xff\xff\xff\xff"
                    newobj = lzma.decompress(buf, lzma.FORMAT_ALONE)
                elif d["_ArrayZipType_"] == "lz4":
                    try:
                        import lz4.frame
                    except ImportError:
                        print(
                            'Warning: you must install "lz4" module to decompress a data record in this file, ignoring'
                        )
                        return copy.deepcopy(d) if opt["inplace"] else d

                    try:
                        newobj = lz4.frame.decompress(bytes(newobj))
                    except Exception as e:
                        raise ValueError(f"lz4 decompression failed: {e}")

                elif d["_ArrayZipType_"].startswith("blosc2"):
                    try:
                        import blosc2
                    except ImportError:
                        print('Warning: you must install "blosc2" module...')
                        return copy.deepcopy(d) if opt["inplace"] else d

                    try:
                        blosc2nthread = 1
                        if "nthread" in opt:
                            blosc2nthread = opt["nthread"]
                        newobj = blosc2.decompress2(
                            bytes(newobj), as_bytearray=False, nthreads=blosc2nthread
                        )
                    except Exception as e:
                        raise ValueError(f"blosc2 decompression failed: {e}")

                newobj = np.frombuffer(bytearray(newobj), dtype=np.dtype(d["_ArrayType_"])).reshape(
                    d["_ArrayZipSize_"]
                )
                # Handle sparse arrays
                if "_ArrayIsSparse_" in d and d["_ArrayIsSparse_"]:
                    try:
                        import scipy.sparse
                    except ImportError:
                        raise ImportError(
                            'To decode sparse JData arrays, install scipy: "pip install scipy"'
                        )
                    shape = (
                        tuple(d["_ArraySize_"])
                        if isinstance(d["_ArraySize_"], list)
                        else (d["_ArraySize_"],)
                    )
                    is_complex = "_ArrayIsComplex_" in d and d["_ArrayIsComplex_"]
                    if isinstance(newobj, np.ndarray):
                        arraydata = newobj
                    else:
                        arraydata = np.array(newobj, dtype=np.float64)
                    if arraydata.ndim == 1:
                        nrows = 4 if is_complex else 3
                        nnz = len(arraydata) // nrows
                        arraydata = arraydata.reshape(nrows, nnz)
                    rows = arraydata[0].astype(np.intp) - 1
                    cols = arraydata[1].astype(np.intp) - 1
                    if is_complex:
                        vals = arraydata[2] + 1j * arraydata[3]
                    else:
                        vals = arraydata[2]
                    newobj = scipy.sparse.csc_matrix((vals, (rows, cols)), shape=shape)
                if "_ArrayIsComplex_" in d and newobj.shape[0] == 2:
                    newobj = newobj[0] + 1j * newobj[1]
                if "_ArrayOrder_" in d and (
                    d["_ArrayOrder_"].lower() == "c"
                    or d["_ArrayOrder_"].lower() == "col"
                    or d["_ArrayOrder_"].lower() == "column"
                ):
                    newobj = newobj.reshape(d["_ArraySize_"], order="F")
                else:
                    newobj = newobj.reshape(d["_ArraySize_"])
                if not hasattr(d["_ArraySize_"], "__iter__") and d["_ArraySize_"] == 1:
                    newobj = newobj.item()
                    return newobj
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
                    d["_ArrayOrder_"].lower() == "c"
                    or d["_ArrayOrder_"].lower() == "col"
                    or d["_ArrayOrder_"].lower() == "column"
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
        elif "_DataLink_" in d:
            if opt["maxlinklevel"] > 0 and "_DataLink_" in d:
                if isinstance(d["_DataLink_"], str):
                    datalink = d["_DataLink_"]
                    if re.search("\:\$", datalink):
                        ref = re.search(
                            "^(?P<proto>[a-zA-Z]+://)*(?P<path>.+)(?P<delim>\:)()*(?P<jsonpath>(?<=:)\$\d*\.*.*)*",
                            datalink,
                        )
                    else:
                        ref = re.search(
                            "^(?P<proto>[a-zA-Z]+://)*(?P<path>.+)(?P<delim>\:)*(?P<jsonpath>(?<=:)\$\d*\..*)*",
                            datalink,
                        )
                    if ref and ref.group("path"):
                        uripath = ref.group("proto") + ref.group("path")
                        newobj, fname = jdlink(uripath)
                        if os.path.exists(fname):
                            opt["maxlinklevel"] = opt["maxlinklevel"] - 1
                            if ref.group("jsonpath"):
                                newobj = jsonpath(newobj, ref.group("jsonpath"))
                        return newobj
                    else:
                        raise Exception(
                            "JData",
                            "_DataLink_ contains invalid URL",
                        )
        return decodedict(d, **opt)
    else:
        return copy.deepcopy(d) if opt["inplace"] else d


##====================================================================================
## helper functions
##====================================================================================


def jsonfilter(obj):
    """
    JSON serialization fallback handler for non-serializable Python types.

    Intended for use as the 'default' parameter of json.dumps(). Converts numpy
    types, bytes, and special floats to JSON-compatible representations.

    Args:
        obj: A Python object that is not natively JSON-serializable.

    Returns:
        A JSON-serializable equivalent: numpy arrays become lists, numpy scalars
        become Python scalars, bytes become UTF-8 strings, NaN/Inf become JData
        string annotations. Returns None if the type is not handled.
    """
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


def encodedict(d0, **kwargs):
    """
    Encode all values in a dict using JData annotations.

    Iterates over key-value pairs and recursively calls encode() on each.
    Keys that are themselves non-string types are also encoded.

    Args:
        d0 (dict): The input dictionary to encode.
        **kwargs: Options passed through to encode() for each value.

    Returns:
        dict: A new dictionary with all values JData-encoded.
    """
    d = dict(d0)
    for k, v in d0.items():
        if isinstance(v, np.ndarray) and isinstance(k, str) and (k in _allownumpy):
            continue
        newkey = encode(k, **kwargs)
        d[newkey] = encode(v, **kwargs)
        if k != newkey:
            d.pop(k)
    return d


# -------------------------------------------------------------------------------------


def encodelist(d0, **kwargs):
    """
    Encode all elements in a list using JData annotations.

    Iterates over list elements and recursively calls encode() on each.

    Args:
        d0 (list): The input list to encode.
        **kwargs: Options passed through to encode() for each element.
            inplace (bool): If True, deep-copy elements before encoding.

    Returns:
        list: A new list with all elements JData-encoded.
    """
    if kwargs.get("inplace", False):
        d = [copy.deepcopy(x) if not isinstance(x, np.ndarray) else x for x in d0]
    else:
        d = list(d0)
    for i, s in enumerate(d):
        d[i] = encode(s, **kwargs)
    return d


# -------------------------------------------------------------------------------------


def decodedict(d0, **kwargs):
    """
    Decode all values in a JData-annotated dict back to native types.

    Iterates over key-value pairs and recursively calls decode() on each value.

    Args:
        d0 (dict): The input JData-annotated dictionary.
        **kwargs: Options passed through to decode() for each value.

    Returns:
        dict: A new dictionary with all values decoded to native Python types.
    """
    d = dict(d0)
    for k, v in d.items():
        newkey = encode(k, **kwargs)
        d[newkey] = decode(v, **kwargs)
        if k != newkey:
            d.pop(k)
    return d


# -------------------------------------------------------------------------------------


def decodelist(d0, **kwargs):
    """
    Decode all elements in a JData-annotated list back to native types.

    Iterates over list elements and recursively calls decode() on each.

    Args:
        d0 (list): The input JData-annotated list.
        **kwargs: Options passed through to decode() for each element.
            inplace (bool): If True, deep-copy elements before decoding.

    Returns:
        list: A new list with all elements decoded to native Python types.
    """
    if kwargs.get("inplace", False):
        d = [copy.deepcopy(x) if not isinstance(x, np.ndarray) else x for x in d0]
    else:
        d = list(d0)
    for i, s in enumerate(d):
        d[i] = decode(s, **kwargs)
    return d


# -------------------------------------------------------------------------------------


def zlibencode(buf):
    """
    Compress a bytes buffer using zlib.

    Args:
        buf (bytes): Raw byte data to compress.

    Returns:
        bytes: Zlib-compressed data.
    """
    return zlib.compress(buf)


# -------------------------------------------------------------------------------------


def gzipencode(buf):
    """
    Compress a bytes buffer using gzip format.

    Args:
        buf (bytes): Raw byte data to compress.

    Returns:
        bytes: Gzip-compressed data.
    """
    gzipper = zlib.compressobj(wbits=(zlib.MAX_WBITS | 16))
    newbuf = gzipper.compress(buf)
    newbuf += gzipper.flush()
    return newbuf


# -------------------------------------------------------------------------------------


def lzmaencode(buf):
    """
    Compress a bytes buffer using LZMA (FORMAT_ALONE).

    Args:
        buf (bytes): Raw byte data to compress.

    Returns:
        bytes: LZMA-compressed data.
    """
    return lzma.compress(buf, lzma.FORMAT_ALONE)


# -------------------------------------------------------------------------------------


def lz4encode(buf):
    """
    Compress a bytes buffer using LZ4 frame format.

    Requires the 'lz4' package to be installed.

    Args:
        buf (bytes): Raw byte data to compress.

    Returns:
        bytes: LZ4-compressed data.

    Raises:
        ImportError: If the lz4 module is not installed.
    """
    try:
        import lz4.frame
    except ImportError:
        raise Exception(
            "JData",
            'you must install "lz4" module to compress with this format',
        )
    return lz4.compress(buf.tobytes(), lzma.FORMAT_ALONE)


# -------------------------------------------------------------------------------------


def base64encode(buf):
    """
    Encode a bytes buffer to base64.

    Args:
        buf (bytes): Raw byte data to encode.

    Returns:
        bytes: Base64-encoded data.
    """
    return base64.b64encode(buf)


# -------------------------------------------------------------------------------------


def zlibdecode(buf):
    """
    Decompress a zlib-compressed bytes buffer.

    Args:
        buf (bytes): Zlib-compressed byte data.

    Returns:
        bytes: Decompressed raw data.
    """
    return zlib.decompress(buf)


# -------------------------------------------------------------------------------------


def gzipdecode(buf):
    """
    Decompress a gzip-compressed bytes buffer.

    Args:
        buf (bytes): Gzip-compressed byte data.

    Returns:
        bytes: Decompressed raw data.
    """
    return zlib.decompress(bytes(buf), zlib.MAX_WBITS | 32)


# -------------------------------------------------------------------------------------


def lzmadecode(buf):
    """
    Decompress an LZMA-compressed bytes buffer.

    Args:
        buf (bytes): LZMA-compressed byte data.

    Returns:
        bytes: Decompressed raw data.
    """
    newbuf = bytearray(buf)  # set length to -1 (unknown) if EOF appears
    newbuf[5:13] = b"\xff\xff\xff\xff\xff\xff\xff\xff"
    return lzma.decompress(newbuf, lzma.FORMAT_ALONE)


# -------------------------------------------------------------------------------------


def lz4decode(buf):
    """
    Decompress an LZ4-compressed bytes buffer.

    Requires the 'lz4' package to be installed.

    Args:
        buf (bytes): LZ4-compressed byte data.

    Returns:
        bytes: Decompressed raw data.

    Raises:
        ImportError: If the lz4 module is not installed.
    """
    try:
        import lz4.frame
    except ImportError:
        raise Exception(
            "JData",
            'you must install "lz4" module to compress with this format',
        )
    return lz4.frame.decompress(bytes(buf))


# -------------------------------------------------------------------------------------


def base64decode(buf):
    """
    Decode a base64-encoded bytes buffer.

    Args:
        buf (bytes): Base64-encoded byte data.

    Returns:
        bytes: Decoded raw data.
    """
    return base64.b64decode(buf)


# -------------------------------------------------------------------------------------


def jdataencode(obj, **kwargs):
    return encode(obj, **kwargs)


# -------------------------------------------------------------------------------------


def jdatadecode(obj, **kwargs):
    return decode(obj, **kwargs)


# -------------------------------------------------------------------------------------
