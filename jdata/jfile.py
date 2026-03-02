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
from .csvtsv import load_csv_tsv, save_csv_tsv
from .jgifti import loadgifti, savegifti
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
    "t": [".json", ".jdt", ".jdat", ".jnii", ".jgii", ".jmsh", ".jnirs", ".jbids"],
    "b": [
        ".ubj",
        ".bjd",
        ".jdb",
        ".jbat",
        ".bnii",
        ".bgii",
        ".bmsh",
        ".pmat",
        ".bnirs",
    ],
    "h5": [".h5", ".hdf5", ".snirf", ".nwb"],
    "nii": [".nii", ".nii.gz", ".img", "img.gz"],
    "gii": [".gii", ".gii.gz"],
    "csv": [".csv", ".csv.gz", ".tsv", "tsv.gz"],
}

##====================================================================================
## Loading and saving data based on file extensions
##====================================================================================


def load(fname, opt=None, **kwargs):
    """
    Load data from a file path or URL with automatic format detection.

    If the input starts with 'http://' or 'https://', it downloads and caches the
    file locally before loading. Otherwise dispatches to loadjd() for local files.

    Args:
        fname (str): A local file path or URL to load data from.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Loading options passed to loadjd() or downloadlink().

    Returns:
        The loaded data structure.
    """
    if opt is None:
        opt = {}
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
    elif ext in jext["gii"]:
        return loadgifti(fname, opt, **kwargs)
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


def save(data, fname, opt=None, **kwargs):
    """
    Save data to a file with automatic format detection. Alias for savejd().

    Args:
        data: The Python data structure to save.
        fname (str): Output file path.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Saving options passed to savejd().

    Returns:
        None
    """
    if opt is None:
        opt = {}
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
    elif ext in jext["gii"]:
        savegifti(data, fname, **kwargs)
    else:
        raise Exception(
            "JData",
            "file extension is not recognized, accept ("
            + ",".join(jext["t"])
            + ";"
            + ",".join(jext["b"])
            + ")",
        )


def loadurl(url, opt=None, **kwargs):
    """
    Load a JData file from a URL without local caching.

    Downloads data from the given URL and parses it directly without saving
    to a local cache file. Use load() instead if caching is desired.

    Args:
        url (str): A URL (http:// or https://) to load data from.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options passed to downloadlink() with nocache=True by default.

    Returns:
        The loaded and decoded data structure.

    Raises:
        Exception: If the URL scheme is not http or https.
    """
    if opt is None:
        opt = {}
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


def loadt(fname, opt=None, **kwargs):
    """
    Load a text-based JSON JData file and decode it to native Python data.

    Reads a JSON file from disk, parses it, and optionally decodes JData
    annotations (arrays, special values) back to native Python/numpy types.

    Args:
        fname (str): Path to a text-based JData file (.json, .jnii, .jdt, etc.).
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            decode (bool): If True (default), call jdata.decode() after loading.
            strict (bool): If False (default), allow non-strict JSON parsing.
            inplace (bool): If True (default), allow in-place decoding.

    Returns:
        The loaded and optionally decoded Python data structure (dict, list, etc.).
    """
    if opt is None:
        opt = {}
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


def savet(data, fname, opt=None, **kwargs):
    """
    Save a Python data structure to a text-based JSON JData file.

    Optionally encodes the data using JData annotations before writing.

    Args:
        data: The Python data structure to save.
        fname (str): Output file path (.json, .jnii, .jdt, etc.).
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            encode (bool): If True (default), call jdata.encode() before saving.
            indent (int): JSON indentation level. None for compact output.
            inplace (bool): If True (default), allow in-place encoding.

    Returns:
        None
    """
    if opt is None:
        opt = {}
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


def loadts(buf, opt=None, **kwargs):
    """
    Parse a JSON string buffer and decode it to native Python data.

    Like loadt() but operates on an in-memory string instead of a file.

    Args:
        buf (str or bytes): A JSON string or byte-stream to parse.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            decode (bool): If True (default), call jdata.decode() after parsing.
            strict (bool): If False (default), allow non-strict JSON parsing.

    Returns:
        The parsed and optionally decoded Python data structure.
    """
    if opt is None:
        opt = {}
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

    data = json.loads(buf, **jsonkwargs)

    if kwargs["decode"]:
        data = decode(data, **kwargs)
    return data


def loadbs(buf, opt=None, **kwargs):
    """
    Parse a binary BJData/UBJSON byte buffer and decode to native Python data.

    Like loadb() but operates on an in-memory buffer instead of a file.
    Requires the 'bjdata' package.

    Args:
        buf (bytes): A BJData/UBJSON byte-buffer to parse.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            decode (bool): If True (default), call jdata.decode() after parsing.

    Returns:
        The parsed and optionally decoded Python data structure.

    Raises:
        ImportError: If the bjdata module is not installed.
    """
    if opt is None:
        opt = {}
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
        data = bjdata.loadb(buf, **bjdkwargs)
        if kwargs["decode"]:
            data = decode(data, **kwargs)
        return data


def show(data, opt=None, **kwargs):
    """
    Print or return a Python data structure as a JSON string.

    Encodes the data with JData annotations and serializes to JSON. Useful for
    debugging and inspecting data structures.

    Args:
        data: The Python data structure to display.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            string (bool): If True, return the JSON string instead of printing.
            encode (bool): If True (default), encode with JData annotations first.
            indent (int): JSON indentation level.

    Returns:
        str or None: The JSON string if string=True, otherwise None (prints to stdout).
    """

    if opt is None:
        opt = {}
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


def dumpb(data, opt=None, **kwargs):
    """
    Serialize a Python data structure to a binary BJData byte stream.

    Requires the 'bjdata' package.

    Args:
        data: The Python data structure to serialize.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options passed through to bjdata.dumpb().

    Returns:
        bytes: The BJData-encoded byte stream.

    Raises:
        ImportError: If the bjdata module is not installed.
    """
    if opt is None:
        opt = {}
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


def loadb(fname, opt=None, **kwargs):
    """
    Load a binary BJData/UBJSON JData file and decode it to native Python data.

    Requires the 'bjdata' package to be installed.

    Args:
        fname (str): Path to a binary JData file (.bjd, .ubj, .bnii, etc.).
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            decode (bool): If True (default), call jdata.decode() after loading.
            islittle (bool): If True, use little-endian byte order (BJData default).

    Returns:
        The loaded and optionally decoded Python data structure.

    Raises:
        ImportError: If the bjdata module is not installed.
    """
    if opt is None:
        opt = {}
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


def saveb(data, fname, opt=None, **kwargs):
    """
    Save a Python data structure to a binary BJData/UBJSON JData file.

    Requires the 'bjdata' package to be installed.

    Args:
        data: The Python data structure to save.
        fname (str): Output file path (.bjd, .ubj, .bnii, etc.).
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            encode (bool): If True (default), call jdata.encode() before saving.
            islittle (bool): If True, use little-endian byte order.

    Returns:
        None

    Raises:
        ImportError: If the bjdata module is not installed.
    """
    if opt is None:
        opt = {}
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


def jsoncache(url, opt=None, **kwargs):
    """
    Compute the local cache folder and filename for a given URL.

    Determines where a downloaded file should be cached based on the URL pattern.
    For NeuroJSON.io URLs, uses a structured folder hierarchy. For other URLs,
    uses SHA-256 hashing of the URL for the filename.

    Args:
        url (str): The URL to compute cache info for.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Additional options.

    Returns:
        tuple: (cachepath, filename) where cachepath is a list of candidate cache
            directories or a single path if the file is already cached, and
            filename is the cache filename.
    """

    if opt is None:
        opt = {}
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

    if (isinstance(url, list) or isinstance(url, tuple) or isinstance(url, frozenset)) and len(
        url
    ) < 4:
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


def jdlink(uripath, opt=None, **kwargs):
    """
    Download and cache externally linked data files referenced by _DataLink_ URLs.

    Processes a single URL or list of URLs (typically extracted via jsonpath from
    a loaded dataset). Downloads files, caches them locally, and optionally parses
    JData-formatted files.

    Args:
        uripath (str or list): A single URL string or a list of URL strings to download.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            regex (str): Filter URLs matching this regular expression pattern.
            showlink (bool): If True, print URLs during download.
            showsize (bool): If True, print total download size.
            downloadonly (bool): If True, download without parsing.
            nocache (bool): If True, re-download ignoring cached files.

    Returns:
        tuple: (data, filename, cachepath) for a single URL, or lists of each
            for multiple URLs.
    """

    if opt is None:
        opt = {}
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
                "total {} links, {} buf, {} files with unknown size".format(
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


def downloadlink(uripath, opt=None, **kwargs):
    """
    Download a single URL and return parsed data with cache path info.

    Low-level function called by jdlink() and load(). Handles downloading,
    caching, and optional parsing of the downloaded content.

    Args:
        uripath (str): The URL to download.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Options including:
            showlink (bool): If True, print URL when downloading.
            nocache (bool): If True, re-download without caching.
            downloadonly (bool): If True, download without parsing.

    Returns:
        tuple: (data, filename, cachepath) where data is the parsed content
            (or None), filename is the local path, and cachepath is the cache info.
    """
    if opt is None:
        opt = {}
    kwargs.setdefault("showlink", 1)
    kwargs.update(opt)

    if kwargs.get("nocache", False) and not kwargs.get("downloadonly", False):
        newdata = urllib.request.urlopen(uripath).read()
        try:
            newdata = loadts(newdata, **kwargs)
        except Exception:
            try:
                newdata = loadbs(newdata, **kwargs)
            except Exception:
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
        if (not kwargs.get("downloadonly", False)) and ext in jext["t"] or ext in jext["b"]:
            newdata = loadjd(fname, **kwargs)

    elif not isinstance(cachepath, list) and os.path.exists(cachepath):
        if kwargs["showlink"]:
            print("loading from cache:", cachepath)
        fname = cachepath
        spl = os.path.splitext(fname)
        ext = spl[1].lower()
        if (not kwargs.get("downloadonly", False)) and ext in jext["t"] or ext in jext["b"]:
            newdata = loadjd(fname, **kwargs)
    return newdata, fname, cachepath


def loadjson(fname, **kwargs):
    """
    Load a text-based JSON file. Alias for loadt().

    Args:
        fname (str): Path to the JSON file.
        **kwargs: Options passed to loadt().

    Returns:
        The loaded data structure.
    """
    return loadt(fname, **kwargs)


def savejson(fname, **kwargs):
    """
    Save data to a text-based JSON file. Alias for savet().

    Args:
        fname (str): Output JSON file path.
        **kwargs: Options passed to savet().

    Returns:
        None
    """
    return savet(fname, **kwargs)


def loadbj(fname, **kwargs):
    """
    Load a binary BJData file. Alias for loadb().

    Args:
        fname (str): Path to the BJData file.
        **kwargs: Options passed to loadb().

    Returns:
        The loaded data structure.
    """
    return loadb(fname, **kwargs)


def savebj(fname, **kwargs):
    """
    Save data to a binary BJData file. Alias for saveb().

    Args:
        fname (str): Output BJData file path.
        **kwargs: Options passed to saveb().

    Returns:
        None
    """
    return saveb(fname, **kwargs)


def loadubjson(*varargin, **kwargs):
    """
    Load a UBJSON file (big-endian BJData). Alias for loadbj() with endian='B'.

    Args:
        *varargin: Positional arguments passed to loadbj().
        **kwargs: Options passed to loadbj().

    Returns:
        The loaded data structure.
    """
    # Set default endian for UBJSON (big-endian)
    kwargs["endian"] = "B"
    return loadbj(*varargin, **kwargs)


def saveubjson(*varargin, **kwargs):
    """
    Save data to a UBJSON file (big-endian BJData). Alias for savebj() with endian='B'.

    Args:
        *varargin: Positional arguments passed to savebj().
        **kwargs: Options passed to savebj().

    Returns:
        None
    """
    # Set default endian for UBJSON (big-endian)
    kwargs["endian"] = "B"
    return savebj(*varargin, **kwargs)


def loadmsgpack(filename: str, **kwargs):
    """
    Load a MessagePack (.msgpack) file into a Python data structure.

    Requires the 'msgpack' package.

    Args:
        filename (str): Path to the MessagePack file.
        **kwargs: Options passed to msgpack.unpack().

    Returns:
        The deserialized Python data structure.

    Raises:
        ValueError: If the msgpack module is not installed.
    """
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
    """
    Save a Python data structure to a MessagePack (.msgpack) file.

    Requires the 'msgpack' package.

    Args:
        data: The Python data structure to save.
        filename (str): Output file path.
        **kwargs: Options passed to msgpack.packb().

    Returns:
        None

    Raises:
        TypeError: If the data cannot be serialized.
        OSError: If the file cannot be written.
    """
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
            if key.lower() in ["elem", "face"] and np.issubdtype(value.dtype, np.integer):
                # Convert 1-based indices to 0-based for Python usage
                data[key] = value - 1
            else:
                # Use F-order for flattening/reshaping
                data[key] = np.array(value, order="F")
    return data


def loadjd(filename: str, suffix=None, **kwargs):
    """
    Load a JData file in any supported format based on file extension.

    Unified loading interface that dispatches to the appropriate loader
    (loadt, loadb, loadh5, loadnifti, loadgifti, load_csv_tsv, etc.)
    based on the file extension.

    Args:
        fname (str): Path to the input file. Supported extensions include
            .json, .jnii, .bjd, .ubj, .bnii, .h5, .snirf, .nii, .nii.gz,
            .gii, .csv, .tsv, .mat, .msgpack, and more.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Format-specific options passed to the underlying loader.

    Returns:
        The loaded data structure (dict, list, numpy array, etc.).

    Raises:
        Exception: If the file extension is not recognized.
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
    """
    Save a Python data structure to a JData file in any supported format.

    Unified saving interface that dispatches to the appropriate saver
    (savet, saveb, saveh5, savenifti, savegifti, etc.) based on file extension.

    Args:
        data: The Python data structure to save.
        fname (str): Output file path. The format is determined by the extension.
        opt (dict, optional): Legacy options dict merged into kwargs.
        **kwargs: Format-specific options passed to the underlying saver.

    Returns:
        None

    Raises:
        Exception: If the file extension is not recognized.
    """
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
    if re.search(r"\.[Ss][Nn][Ii][Rr][Ff]$", filename) or re.search(r"\.[Hh]5$", filename):
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
    elif re.search(r"\.[Ss][Nn][Ii][Rr][Ff]$", filename) or re.search(r"\.[Hh]5$", filename):
        save(jnirs, filename, **kwargs)
    else:
        raise ValueError("file suffix must be .jnirs for text JSNIRF or .bnirs for binary JSNIRF")


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
