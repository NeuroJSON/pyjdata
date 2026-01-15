"""@package docstring
File IO to load/decode JData-based files to Python data or encode/save Python data to JData files

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "loadjson",
    "savejson",
    "loadbj",
    "savebj",
    "loadjd",
    "savejd",
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
    "jsoncache",
    "jdlink",
    "jext",
    "loadjsnirf",
    "loadsnirf",
    "savejsnirf",
    "savesnirf",
    "loadmsgpack",
    "savemsgpack",
    "loadubjson",
    "saveubjson",
]

##====================================================================================
## dependent libraries
##====================================================================================

import json
import os
import re
import numpy as np
from .jdata import encode, decode, jsonfilter
from .h5 import loadh5, saveh5, snirfdecode, soa2aos
from .jnifti import loadnifti, loadjnifti, savenifti
from .csv import load_csv_tsv, save_csv_tsv
import urllib.request
from hashlib import sha256
from sys import platform
from collections import OrderedDict
from typing import Dict
import warnings

##====================================================================================
## global variables
##====================================================================================

jext = {
    "t": [".json", ".jdt", ".jdat", ".jnii", ".jmsh", ".jnirs", ".jbids"],
    "b": [".ubj", ".bjd", ".jdb", ".jbat", ".bnii", ".bmsh", ".pmat", ".bnirs"],
    "h5": [".h5", ".hdf5", ".snirf", ".nwb"],
    "nii": [".nii", ".nii.gz", ".img", "img.gz"],
    "csv": [".csv", ".csv.gz", ".tsv", "tsv.gz"],
}

##====================================================================================
## Loading and saving data based on file extensions
##====================================================================================


def load(fname, opt={}, **kwargs):
    """@brief Loading a JData file (binary or text) according to the file extension

    @param[in] fname: a JData file name (accept .json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    if re.match("^https*://", fname):
        newdata, fname, _ = downloadlink(fname, opt, **kwargs)
        if newdata:
            return newdata[0]

    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if ext in jext["t"]:
        return loadt(fname, opt, **kwargs)
    elif ext in jext["b"]:
        return loadb(fname, opt, **kwargs)
    elif ext in jext["nii"]:
        return loadjnifti(fname, opt, **kwargs)
    elif ext in jext["csv"]:
        return load_csv_tsv(fname, **kwargs)
    else:
        raise Exception(
            "JData",
            "file extension is not recognized, accept ("
            + ",".join(jext["t"])
            + ";"
            + ",".join(jext["b"])
            + ")",
        )


def save(data, fname, opt={}, **kwargs):
    """@brief Saving Python data to file (binary or text) according to the file extension

    @param[in] data: data to be saved
    @param[in] fname: a JData file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if ext == "ubj":
        kwargs.setdefault("islittle", False)

    if ext in jext["t"]:
        savet(data, fname, opt, **kwargs)
    elif ext in jext["b"]:
        saveb(data, fname, opt, **kwargs)
    elif ext == ".snirf":
        savesnirf(data, fname, **kwargs)
    elif ext in jext["h5"]:
        saveh5(data, fname, **kwargs)
    elif ext in jext["nii"]:
        savenifti(data, fname, **kwargs)
    else:
        raise Exception(
            "JData",
            "file extension is not recognized, accept ("
            + ",".join(jext["t"])
            + ";"
            + ",".join(jext["b"])
            + ")",
        )


def loadurl(url, opt={}, **kwargs):
    """@brief Loading a JData file (binary or text) from a URL without caching locally

    @param[in] url: a REST API URL, curently only support http:// and https://
    @param[in] opt: options, opt['nocache']=True by default, setting to False download and locally cache the data
    """
    kwargs.setdefault("nocache", True)
    kwargs.update(opt)

    if re.match("^https*://", url):
        newdata = downloadlink(url, **kwargs)
        return newdata[0]
    else:
        raise Exception(
            "JData",
            "input to loadurl is not a valid URL",
        )


##====================================================================================
## Loading and saving text-based JData (i.e. JSON) files
##====================================================================================


def loadt(fname, opt={}, **kwargs):
    """@brief Loading a text-based (JSON) JData file and decode it to native Python data

    @param[in] fname: a text JData (JSON based) file name
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    kwargs.setdefault("strict", False)
    kwargs.setdefault("object_pairs_hook", OrderedDict)
    kwargs.setdefault("decode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = True
    kwargs.update(opt)

    jsonkwargs = {
        k: kwargs[k]
        for k in (
            "parse_float",
            "parse_int",
            "parse_constant",
            "strict",
            "object_pairs_hook",
        )
        if k in kwargs
    }

    with open(fname, "r") as fid:
        data = json.load(fid, **jsonkwargs)

    if kwargs["decode"]:
        data = decode(data, **kwargs)
    return data


def savet(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a text-based JData (JSON) file

    @param[in] data: data to be saved
    @param[in] fname: a text JData (JSON based) file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    kwargs.setdefault("default", jsonfilter)
    kwargs.setdefault("encode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = True
    kwargs.update(opt)

    jsonkwargs = {
        k: kwargs[k]
        for k in (
            "indent",
            "separators",
            "allow_nan",
            "default",
            "sort_keys",
            "skipkeys",
        )
        if k in kwargs
    }

    if kwargs["encode"]:
        data = encode(data, **kwargs)

    with open(fname, "w") as fid:
        json.dump(data, fid, **jsonkwargs)


##====================================================================================
## In-memory buffer Parse and dump
##====================================================================================


def loadts(bytes, opt={}, **kwargs):
    """@brief Loading a text-based (JSON) JData string buffer and decode it to native Python data

    @param[in] bytes: a JSON string or byte-stream
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    kwargs.setdefault("strict", False)
    kwargs.setdefault("object_pairs_hook", OrderedDict)
    kwargs.setdefault("decode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = True
    kwargs.update(opt)

    jsonkwargs = {
        k: kwargs[k]
        for k in (
            "parse_float",
            "parse_int",
            "parse_constant",
            "strict",
            "object_pairs_hook",
        )
        if k in kwargs
    }

    data = json.loads(bytes, **jsonkwargs)

    if kwargs["decode"]:
        data = decode(data, **kwargs)
    return data


def loadbs(bytes, opt={}, **kwargs):
    """@brief Loading a binary-JSON/BJData string buffer and decode it to native Python data

    @param[in] bytes: a BJData byte-buffer or byte-stream
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    kwargs.setdefault("decode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = False
    kwargs.update(opt)

    bjdkwargs = {
        k: kwargs[k]
        for k in (
            "no_bytes",
            "uint8_bytes",
            "object_hook",
            "intern_object_keys",
            "object_pairs_hook",
            "islittle",
        )
        if k in kwargs
    }

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        data = bjdata.loadb(bytes, **bjdkwargs)
        if kwargs["decode"]:
            data = decode(data, **kwargs)
        return data


def show(data, opt={}, **kwargs):
    """@brief Printing a python data as JSON string or return the JSON string (opt['string']=True)

    @param[in] data: data to be saved
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before printing
    """

    kwargs.setdefault("default", jsonfilter)
    kwargs.setdefault("string", False)
    kwargs.setdefault("encode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = True
    kwargs.update(opt)

    jsonkwargs = {
        k: kwargs[k]
        for k in (
            "indent",
            "separators",
            "allow_nan",
            "default",
            "sort_keys",
            "skipkeys",
        )
        if k in kwargs
    }

    if kwargs["encode"]:
        data = encode(data, **kwargs)

    str = json.dumps(data, **jsonkwargs)

    if kwargs["string"]:
        return str
    else:
        print(str)


def dumpb(data, opt={}, **kwargs):
    """@brief Printing native python data in binary JSON stream

    @param[in] data: data to be saved
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before printing
    """
    kwargs.update(opt)

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        return bjdata.dumpb(data, **kwargs)


##====================================================================================
## Loading and saving binary JData (i.e. UBJSON) files
##====================================================================================


def loadb(fname, opt={}, **kwargs):
    """@brief Loading a binary (BJData/UBJSON) JData file and decode it to native Python data

    @param[in] fname: a binary (BJData/UBJSON based) JData file name
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    """
    kwargs.setdefault("decode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = False
    kwargs.update(opt)

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        bjdkwargs = {
            k: kwargs[k]
            for k in (
                "no_bytes",
                "uint8_bytes",
                "object_hook",
                "intern_object_keys",
                "object_pairs_hook",
                "islittle",
            )
            if k in kwargs
        }

        with open(fname, "rb") as fid:
            data = bjdata.load(fid, **bjdkwargs)
        if kwargs["decode"]:
            data = decode(data, **kwargs)
        return data

    return None


def saveb(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a binary JData (BJData/UBJSON) file

    @param[in] data: data to be saved
    @param[in] fname: a binary (BJData/UBJSON based) JData file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    kwargs.setdefault("encode", True)
    kwargs.setdefault("inplace", True)
    kwargs["base64"] = False
    kwargs.update(opt)

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        bjdkwargs = {
            k: kwargs[k]
            for k in (
                "container_count",
                "sort_keys",
                "no_float32",
                "uint8_bytes",
                "default",
                "islittle",
            )
            if k in kwargs
        }
        if kwargs["encode"]:
            data = encode(data, **kwargs)
        with open(fname, "wb") as fid:
            bjdata.dump(data, fid, **bjdkwargs)


##====================================================================================
## Handling externally linked data files
##====================================================================================


def jsoncache(url, opt={}, **kwargs):
    """@brief Printing the local folder and file name where a linked data file in the URL to be saved

    @param[in] url: a URL
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    """

    pathname = os.getenv("HOME")
    cachepath = [os.path.join(os.getcwd(), ".neurojson")]
    dbname, docname, filename = None, None, None

    if pathname != os.getcwd():
        cachepath.append(os.path.join(pathname, ".neurojson"))

    if platform == "win32":
        cachepath.append(os.path.join(os.getenv("PROGRAMDATA"), "neurojson"))
    elif platform == "darwin":
        cachepath.append(os.path.join(pathname, "Library/neurojson"))
        cachepath.append("/Library/neurojson")
    else:
        cachepath.append(os.path.join(pathname, ".cache/neurojson"))
        cachepath.append("/var/cache/neurojson")

    if (
        isinstance(url, list) or isinstance(url, tuple) or isinstance(url, frozenset)
    ) and len(url) < 4:
        domain = "default"

    if isinstance(url, str):
        link = url
        if re.match("^file://", link) or not re.search("://", link):
            filename = re.sub("^file://", "", link)
            if os.path.isfile(filename):
                cachepath = filename
                filename = True
                return cachepath, filename

        else:
            if re.match("^https*://neurojson.org/io/", link):
                domain = "io"
            else:
                newdomain = re.sub("^(https*|ftp)://([^\/?#:]+).*$", r"\2", link)
                if newdomain:
                    domain = newdomain

            dbname = re.search("(?<=db=)[^&]+", link)
            docname = re.search("(?<=doc=)[^&]+", link)
            filename = re.search("(?<=file=)[^&]+", link)
            if dbname:
                dbname = dbname.group(0)
            if docname:
                docname = docname.group(0)
            if filename:
                filename = filename.group(0)

            if not filename and domain == "neurojson.io":
                ref = re.search(
                    "^(https*|ftp)://neurojson.io(:\d+)*(?P<dbname>/[^\/]+)(?P<docname>/[^\/]+)(?P<filename>/[^\/?]+)*",
                    link,
                )
                if ref:
                    if ref.group("dbname"):
                        dbname = ref.group("dbname")[1:]
                    if ref.group("docname"):
                        docname = ref.group("docname")[1:]
                    if ref.group("filename"):
                        filename = ref.group("filename")[1:]
                    elif dbname:
                        if docname:
                            filename = docname + ".json"
                        else:
                            filename = dbname + ".json"

            if not filename:
                filename = sha256(link.encode("utf-8")).hexdigest()
                suffix = re.search("((\.\w{1,5})+)(?=([#&].*)*$)", link)
                if not suffix:
                    suffix = ""
                else:
                    suffix = suffix.group(0)
                filename = filename + suffix
                if not dbname:
                    dbname = filename[0:2]
                if not docname:
                    docname = filename[2:4]

    p = globals().get("NEUROJSON_CACHE")
    if isinstance(url, str) or (
        isinstance(url, list)
        or isinstance(url, tuple)
        or isinstance(url, frozenset)
        and len(url) >= 3
    ):
        if p is not None:
            cachepath.insert(0, p)
        elif dbname and docname:
            cachepath = [os.path.join(x, domain, dbname, docname) for x in cachepath]
        if filename is not None:
            for i in range(len(cachepath)):
                if os.path.exists(os.path.join(cachepath[i], filename)):
                    cachepath = os.path.join(cachepath[i], filename)
                    filename = True
                    return cachepath, filename
        elif "link" in locals():
            filename = os.path.basename(link)
        if p is not None:
            cachepath.pop(1)
        else:
            cachepath.pop(0)
        return cachepath, filename


def jdlink(uripath, opt={}, **kwargs):
    """@brief Printing the local folder and file name where a linked data file in the URL to be saved

    newdata, fname, cachepath = jdlink(uripath, showlink=True, showsize=True, regex=None, nocache=False, **kwargs)

    @param[in] uripath: a URL
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    showlink: print URL
    showsize: print file size if URL contains the size info
    downloadonly: only download the file, do not parse
    regex: a regular expression string, used to filter uripath list
    nocache: redownload and ignore the cached files
    """

    kwargs.setdefault("showlink", 1)
    kwargs.setdefault("showsize", 1)
    kwargs.update(opt)

    if isinstance(uripath, list):
        if "regex" in kwargs:
            pat = re.compile(kwargs["regex"])
            uripath = [uri for uri in uripath if pat.search(uri)]
            print(uripath)
        if "showsize" in kwargs:
            totalsize = 0
            nosize = 0
            for i in range(len(uripath)):
                filesize = re.findall(r"&size=(\d+)", uripath[i])
                if filesize and filesize[0]:
                    totalsize += int(filesize[0])
                else:
                    nosize += 1
            print(
                "total {} links, {} bytes, {} files with unknown size".format(
                    len(uripath), totalsize, nosize
                )
            )
        alloutput = [[] for _ in range(3)]
        for i in range(len(uripath)):
            newdata, fname, cachepath = downloadlink(uripath[i], **kwargs)
            alloutput[0].append(newdata)
            alloutput[1].append(fname)
            alloutput[2].append(cachepath)
        if len(uripath) == 1:
            alloutput = [x[0] for x in alloutput]
        newdata, fname, cachepath = tuple(alloutput)
    elif isinstance(uripath, str):
        newdata, fname, cachepath = downloadlink(uripath, **kwargs)
    return newdata, fname


def downloadlink(uripath, opt={}, **kwargs):
    """
    newdata, fname, cachepath = downloadlink(urlpath, showlink=True, nocache=False, **kwargs)

    @param[in] uripath: a list of URLs or a single URL as a string
    @param[in] showlink: print the URL when downloading each file
    @param[in] nocache: when True, redownload the data instead of using locally cached files
    kwargs: additional parameters passing to loadjd()
    """
    kwargs.setdefault("showlink", 1)
    kwargs.update(opt)

    if kwargs.get("nocache", False) and not kwargs.get("downloadonly", False):
        newdata = urllib.request.urlopen(uripath).read()
        try:
            newdata = loadts(newdata, **kwargs)
        except:
            try:
                newdata = loadbs(newdata, **kwargs)
            except:
                pass
        return newdata, uripath, None

    newdata = None
    cachepath, filename = jsoncache(uripath)
    if isinstance(cachepath, list) and cachepath:
        if kwargs["showlink"]:
            print("downloading from URL:", uripath)
        fname = os.path.join(cachepath[0], filename)
        fpath = os.path.dirname(fname)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        rawdata = urllib.request.urlopen(uripath).read()
        with open(fname, "wb") as fid:
            fid.write(rawdata)
        spl = os.path.splitext(fname)
        ext = spl[1].lower()
        if kwargs.get("downloadonly", False) and ext in jext["t"] or ext in jext["b"]:
            newdata = loadjd(fname, **kwargs)

    elif not isinstance(cachepath, list) and os.path.exists(cachepath):
        if kwargs["showlink"]:
            print("loading from cache:", cachepath)
        fname = cachepath
        spl = os.path.splitext(fname)
        ext = spl[1].lower()
        if kwargs.get("downloadonly", False) and ext in jext["t"] or ext in jext["b"]:
            newdata = loadjd(fname, **kwargs)
    return newdata, fname, cachepath


def loadjson(fname, **kwargs):
    return loadt(fname, **kwargs)


def savejson(fname, **kwargs):
    return savet(fname, **kwargs)


def loadbj(fname, **kwargs):
    return loadb(fname, **kwargs)


def savebj(fname, **kwargs):
    return saveb(fname, **kwargs)


def loadubjson(*varargin, **kwargs):
    # Set default endian for UBJSON (big-endian)
    kwargs["endian"] = "B"
    return loadbj(*varargin, **kwargs)


def saveubjson(*varargin, **kwargs):
    # Set default endian for UBJSON (big-endian)
    kwargs["endian"] = "B"
    return savebj(*varargin, **kwargs)


def loadmsgpack(filename: str, **kwargs):
    """Load MessagePack files"""
    try:
        import msgpack

        with open(filename, "rb") as f:
            data = msgpack.unpack(f, raw=False, **kwargs)
        return data
    except ImportError:
        raise ValueError(
            f"Failed to load MessagePack file. Install msgpack library for MessagePack support."
        )


def savemsgpack(data, filename: str, **kwargs):
    """Save MessagePack files"""
    try:
        import msgpack

        packed_data = msgpack.packb(data, **kwargs)

        with open(filename, "wb") as f:
            f.write(packed_data)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Failed to serialize data: {e}")
    except OSError as e:
        raise OSError(f"Failed to write file '{filename}': {e}")


def loadmat(filename, **kwargs):
    # Load MATLAB .mat file
    from scipy.io import loadmat as loadmatlab

    data = loadmatlab(filename, **kwargs)

    # Remove MATLAB metadata keys
    matlab_keys = ["__header__", "__version__", "__globals__"]
    for key in matlab_keys:
        data.pop(key, None)

    # Convert arrays to F-order
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Handle 1-based indices: if array contains integers that could be indices (elem/face)
            if key.lower() in ["elem", "face"] and np.issubdtype(
                value.dtype, np.integer
            ):
                # Convert 1-based indices to 0-based for Python usage
                data[key] = value - 1
            else:
                # Use F-order for flattening/reshaping
                data[key] = np.array(value, order="F")
    return data


def loadjd(filename: str, suffix=None, **kwargs):
    """
    Parse a hierarchical container data file, including JSON,
    binary JSON (BJData/UBJSON/MessagePack) and HDF5, and output
    the parsed data in a Python data structure

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        filename: the input hierarchical container data file, supporting:
                *.json,.jnii,.jdt,.jmsh,.jnirs,.jbids: JSON/JData based data files, see https://neurojson.org/jdata/draft3
                *.bjd,.bnii,.jdb,.bmsh,.bnirs,.pmat: binary JData (BJData) files, see https://neurojson.org/bjdata/draft3
                *.ubj: UBJSON-encoded files, see http://ubjson.org
                *.msgpack: MessagePack-encoded files, see http://msgpack.org
                *.h5,.hdf5,.snirf,.nwb: HDF5 files, handled in jdata.h5
                *.nii,.nii.gz,.img,.img.gz: NIfTI files, handled in jdata.jnifti
                *.tsv,.tsv.gz,.csv,.csv.gz: TSV/CSV files, handled in jdata.csv
                *.bval,.bvec: EEG .bval and .bvec files
                *.mat: MATLAB/Octave .mat files
        **varargin: optional keyword arguments for different file types:
                - for JSON/JData files, these are optional parameters supported by jdata.load()
                - for BJData/UBJSON/MessagePack files, these are options supported by jdata.load()
                - for HDF5 files, these are options supported by jdata.loadh5()

    Returns:
        data: a dictionary structure (array) or list (array) storing the hierarchical data
              in the container data file
        mmap: (optional) output storing the JSON/binary JSON memory-map table for fast
              disk access. Available for JSON/binary JSON files when requested.

    Examples:
        obj = {'string': 'value', 'array': [1, 2, 3]}
        jd.save(obj, 'datafile.json')
        newobj = jd.loadjd('datafile.json')

    License:
        BSD or GPL version 3, see LICENSE_{BSD,GPLv3}.txt files for details

    This function is part of JSONLab toolbox (http://iso2mesh.sf.net/cgi-bin/index.cgi?jsonlab)
    """

    if not filename:
        raise ValueError("you must provide file name")

    if re.match("^https*://", filename):
        kwargs.setdefault("downloadonly", True)
        newdata, filename = jdlink(filename, **kwargs)
        if newdata is not None:
            return newdata

    # Convert filename to lowercase for pattern matching
    fname, fileext = os.path.splitext(filename)
    fileext = fileext.lower()

    if suffix:
        fileext = suffix

    if fileext == ".gz":
        fileext = os.path.splitext(fname)[-1].lower() + fileext

    # JSON/JData files
    if fileext in jext["t"]:
        return loadjson(filename, **kwargs)

    # Binary JData files
    elif fileext in jext["b"]:
        return loadbj(filename, **kwargs)

    # UBJSON files
    elif fileext == ".ubj":
        return loadubjson(filename, **kwargs)

    # MessagePack files
    elif fileext == ".msgpack":
        return loadmsgpack(filename, **kwargs)

    # HDF5 based .snirf files
    elif fileext == ".snirf":
        return loadsnirf(filename, **kwargs)

    # HDF5 files
    elif fileext in (".h5", ".hdf5"):
        return loadh5(filename, **kwargs)

    # NIfTI files
    elif fileext in jext["nii"]:
        return loadnifti(filename, **kwargs)

    # TSV/CSV files
    elif fileext in jext["csv"]:
        return load_csv_tsv(filename, **kwargs)

    # MATLAB .mat files and EEG files
    elif fileext == ".mat":
        return loadmat(filename, **kwargs)

    elif fileext in (".bvec", ".bval"):
        return np.loadtxt(filename, **kwargs)

    # Unsupported format - load as raw text
    else:
        warnings.warn(
            "only support parsing .json,.jnii,.jdt,.jmsh,.jnirs,.jbids,.bjd,.bnii,.jdb,.bmsh,.bnirs,.ubj,.msgpack,.h5,.hdf5,.snirf,.pmat,.nwb,.nii,.nii.gz,.tsv,.tsv.gz,.csv,.csv.gz,.mat,.bvec,.bval; load unparsed raw data"
        )
        return None


def savejd(data, filename, suffix=None, **kwargs):
    if not filename:
        raise ValueError("you must provide file name")

    # Convert filename to lowercase
    fname, fileext = os.path.splitext(filename)
    fileext = fileext.lower()

    if fileext == ".gz":
        fileext = os.path.splitext(fname)[-1].lower() + fileext

    if suffix:
        fileext = suffix

    # JSON/JData files
    if fileext in jext["t"]:
        return save(data, filename, **kwargs)

    # UBJSON files
    elif fileext == ".ubj":
        return saveubjson(data, filename, **kwargs)

    # UBJSON files
    elif fileext == ".msgpack":
        return savemsgpack(data, filename, **kwargs)

    # HDF5 based .snirf files
    elif fileext == ".snirf":
        return savesnirf(data, filename, **kwargs)

    # TSV/CSV files
    elif fileext in jext["csv"]:
        return save_csv_tsv(data, filename, **kwargs)

    # HDF5 files
    elif fileext in (".h5", ".hdf5"):
        return saveh5(data, filename, **kwargs)

    # NIfTI files
    elif fileext in jext["nii"]:
        return savenifti(data, filename, **kwargs)

    elif fileext in (".bvec", ".bval"):
        return np.savetxt(data, filename, **kwargs)

    # Binary JData files
    elif fileext in jext["b"]:
        return save(data, filename, **kwargs)

    # Unsupported format - load as raw text
    else:
        warnings.warn(
            "only support parsing .json,.jnii,.jdt,.jmsh,.jnirs,.jbids,.bjd,.bnii,.jdb,.bmsh,.bnirs,.ubj,.msgpack,.h5,.hdf5,.snirf,.pmat,.nwb,.nii,.nii.gz,.tsv,.tsv.gz,.csv,.csv.gz,.mat,.bvec,.bval; load unparsed raw data"
        )
        return None


def loadjsnirf(filename: str, **kwargs) -> Dict:
    """
    Load a text (.jnirs or .json) or binary (.bnirs) based JSNIRF
    file defined in the JSNIRF specification:
    https://github.com/NeuroJSON/jsnirfy or a .snirf/.h5 SNIRF data defined in
    the SNIRF specification https://github.com/fNIRS/snirf

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        filename: the input file name to the JSNIRF or SNIRF file
                *.bnirs for binary JSNIRF file
                *.jnirs for text JSNIRF file
                *.snirf for HDF5/SNIRF files
        **kwargs: optional parameters for loading

    Returns:
        jnirs: a dictionary structure containing the loaded data

    Example:
        newjnirs = loadjsnirf('subject1.jnirs')

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if not filename:
        raise ValueError("you must provide data and output file name")

    # Check file extension and load accordingly
    if re.search(r"\.[Ss][Nn][Ii][Rr][Ff]$", filename) or re.search(
        r"\.[Hh]5$", filename
    ):
        jnirs = loadsnirf(filename, **kwargs)
    elif re.search(r"\.[Jj][Nn][Ii][Rr][Ss]$", filename):
        jnirs = load(filename, **kwargs)
    elif re.search(r"\.[Bb][Nn][Ii][Rr][Ss]$", filename):
        jnirs = load(filename, **kwargs)
    else:
        raise ValueError(
            "file suffix must be .snirf for SNIRF/HDF5, .jnirs for text JSNIRF, .bnirs for binary JSNIRF files"
        )

    return jnirs


def loadsnirf(fname: str, **kwargs) -> Dict:
    """
    Load an HDF5 based SNIRF file, and optionally convert it to a JSON
    file based on the JSNIRF specification:
    https://github.com/NeuroJSON/jsnirfy

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        fname: the input snirf data file name (HDF5 based)
        **kwargs: optional parameters

    Returns:
        data: a dictionary structure with the grouped data fields

    Example:
        data = loadsnirf('test.snirf')

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if not fname or not isinstance(fname, str):
        raise ValueError("you must provide a file name")

    data = loadh5(fname, stringarray=True)[0]

    # Apply snirfdecode based on arguments
    if len(kwargs) == 1 and "format" not in kwargs:
        # Single non-keyword argument case
        format_arg = list(kwargs.values())[0]
        data = snirfdecode(data, format_arg)
    else:
        data = snirfdecode(data, **kwargs)

    # Handle output file if specified
    outfile = kwargs.get("filename", "")
    if outfile:
        if re.search(r"\.[Bb][Nn][Ii][Rr][Ss]$", outfile):
            save({"SNIRFData": data}, outfile, **kwargs)
        elif re.search(r"\.[Jj][Nn][Ii][Rr][Ss]$", outfile) or re.search(
            r"\.[Jj][Ss][Oo][Nn]$", outfile
        ):
            save({"SNIRFData": data}, outfile, **kwargs)
        elif re.search(r"\.[Mm][Aa][Tt]$", outfile):
            # For .mat files, would need scipy.io.savemat
            pass
        else:
            raise ValueError("only support .jnirs,.bnirs and .mat files")

    return data


def savejsnirf(jnirs: Dict, filename: str, **kwargs):
    """
    Save an in-memory JSNIRF structure into a JSNIRF file with format
    defined in JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        jnirs: a dictionary structure containing JSNIRF data
        filename: the output file name to the JSNIRF file
                *.bnirs for binary JSNIRF file
                *.jnirs for text JSNIRF file
                *.snirf or *.h5 for HDF5-based SNIRF file
        **kwargs: optional parameters for saving

    Example:
        jnirs = jsnirfcreate(aux={'name': 'pO2', 'dataTimeSeries': list(range(1,11)), 'time': list(range(1,11))})
        savejsnirf(jnirs, 'test.jnirs')

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if not jnirs or not filename:
        raise ValueError("you must provide data and output file name")

    if re.search(r"\.[Jj][Nn][Ii][Rr][Ss]$", filename):
        save(jnirs, filename, **kwargs)
    elif re.search(r"\.[Bb][Nn][Ii][Rr][Ss]$", filename):
        save(jnirs, filename, **kwargs)
    elif re.search(r"\.[Ss][Nn][Ii][Rr][Ff]$", filename) or re.search(
        r"\.[Hh]5$", filename
    ):
        save(jnirs, filename, **kwargs)
    else:
        raise ValueError(
            "file suffix must be .jnirs for text JSNIRF or .bnirs for binary JSNIRF"
        )


def savesnirf(data: Dict, outfile: str, **kwargs):
    """
    Save SNIRF data to HDF5 based SNIRF file, and optionally convert it to a JSON
    file based on the JSNIRF specification:
    https://github.com/NeuroJSON/jsnirfy

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        data: a raw SNIRF data, preprocessed SNIRF data or JSNIRF data
        outfile: the output SNIRF (.snirf) or JSNIRF data file name (.jnirs, .bnirs)
        **kwargs: optional parameters

    Example:
        data = loadsnirf('test.snirf')
        savesnirf(data, 'newfile.snirf')

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if not data or not outfile or not isinstance(outfile, str):
        raise ValueError("you must provide data and a file name")

    # Set default options
    kwargs.setdefault("rootname", "")
    kwargs.setdefault("variablelengthstring", 1)
    kwargs.setdefault("rowas1d", 1)

    # Convert JSNIRF to SNIRF format if needed
    if "SNIRFData" in data:
        data["nirs"] = data["SNIRFData"].copy()
        if "formatVersion" in data["SNIRFData"]:
            data["formatVersion"] = data["SNIRFData"]["formatVersion"]
            del data["nirs"]["formatVersion"]
        del data["SNIRFData"]

    if outfile:
        if re.search(r"\.[Hh]5$", outfile):
            save(data, outfile, **kwargs)
        elif re.search(r"\.[Ss][Nn][Ii][Rr][Ff]$", outfile):
            # Handle measurementList conversion
            if (
                "nirs" in data
                and "data" in data["nirs"]
                and "measurementList" in data["nirs"]["data"]
            ):
                ml = data["nirs"]["data"]["measurementList"]
                if (
                    isinstance(ml, dict)
                    and "sourceIndex" in ml
                    and hasattr(ml["sourceIndex"], "__len__")
                    and len(ml["sourceIndex"]) > 1
                ):
                    data["nirs"]["data"]["measurementList"] = soa2aos(ml)

                # Force integer types for specific fields
                forceint = [
                    "sourceIndex",
                    "detectorIndex",
                    "wavelengthIndex",
                    "dataType",
                    "dataTypeIndex",
                    "moduleIndex",
                    "sourceModuleIndex",
                    "detectorModuleIndex",
                ]

                ml_list = data["nirs"]["data"]["measurementList"]
                if not isinstance(ml_list, list):
                    ml_list = [ml_list]

                for j, ml_item in enumerate(ml_list):
                    for field in forceint:
                        if field in ml_item:
                            if isinstance(ml_item[field], list):
                                ml_item[field] = np.array(ml_item[field])
                            ml_item[field] = np.int32(ml_item[field])

            # Force indexing for SNIRF format
            data["nirs"] = _forceindex(data["nirs"], "data")
            data["nirs"] = _forceindex(data["nirs"], "stim")
            data["nirs"] = _forceindex(data["nirs"], "aux")

            save(data, outfile, **kwargs)

        elif re.search(r"\.[Jj][Nn][Ii][Rr][Ss]$", outfile) or re.search(
            r"\.[Jj][Ss][Oo][Nn]$", outfile
        ):
            save({"SNIRFData": data}, outfile, **kwargs)
        elif re.search(r"\.[Bb][Nn][Ii][Rr][Ss]$", outfile):
            save({"SNIRFData": data}, outfile, **kwargs)
        elif re.search(r"\.[Mm][Aa][Tt]$", outfile):
            # For .mat files, would need scipy.io.savemat
            pass
        else:
            raise ValueError("only support .snirf, .h5, .jnirs, .bnirs and .mat files")


def _forceindex(root: Dict, name: str) -> Dict:
    """
    Force adding index 1 to the group name for singular struct and cell

    Args:
        root: dictionary to modify
        name: field name to check and potentially rename

    Returns:
        Modified dictionary with indexed field names
    """
    newroot = root.copy()

    if name in newroot and not isinstance(newroot[name], list):
        # Single item - add index 1
        indexed_name = f"{name}1"
        newroot[indexed_name] = newroot[name]
        del newroot[name]

        # Maintain field order
        fields = list(newroot.keys())
        # Move indexed field to original position
        if indexed_name in fields:
            fields.remove(indexed_name)
            # Find where name would have been
            orig_pos = len(fields)  # Default to end if not found
            fields.insert(orig_pos, indexed_name)

        # Rebuild ordered dictionary
        ordered_root = OrderedDict()
        for field in fields:
            ordered_root[field] = newroot[field]
        newroot = ordered_root

    return newroot


def loadrawfile(filename: str) -> str:
    """Load file as raw text"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try binary mode if UTF-8 fails
        try:
            with open(filename, "rb") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file '{filename}': {str(e)}")
