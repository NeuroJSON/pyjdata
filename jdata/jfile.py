"""@package docstring
File IO to load/decode JData-based files to Python data or encode/save Python data to JData files

Copyright (c) 2019-2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
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
]

##====================================================================================
## dependent libraries
##====================================================================================

import json
import os
import re
import jdata as jd
import urllib.request
from hashlib import sha256
from sys import platform
from collections import OrderedDict

##====================================================================================
## global variables
##====================================================================================

jext = {
    "t": [".json", ".jdt", ".jdat", ".jnii", ".jmsh", ".jnirs", ".jbids"],
    "b": [".ubj", ".bjd", ".jdb", ".jbat", ".bnii", ".bmsh", ".pmat", ".bnirs"],
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
        newdata = downloadlink(fname, opt, **kwargs)
        return newdata[0]

    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if ext in jext["t"]:
        return loadt(fname, opt, **kwargs)
    elif ext in jext["b"]:
        return loadb(fname, opt, **kwargs)
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
    opt.setdefault("nocache", True)

    if re.match("^https*://", url):
        newdata = downloadlink(url, opt, **kwargs)
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
    opt.setdefault("decode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = True

    with open(fname, "r") as fid:
        data = json.load(fid, **kwargs)

    if opt["decode"]:
        data = jd.decode(data, opt)
    return data


def savet(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a text-based JData (JSON) file

    @param[in] data: data to be saved
    @param[in] fname: a text JData (JSON based) file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    kwargs.setdefault("default", jd.jsonfilter)
    opt.setdefault("encode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = True

    if opt["encode"]:
        data = jd.encode(data, opt)

    with open(fname, "w") as fid:
        json.dump(data, fid, **kwargs)


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
    opt.setdefault("decode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = True

    data = json.loads(bytes, **kwargs)

    if opt["decode"]:
        data = jd.decode(data, opt)
    return data


def loadbs(bytes, opt={}, **kwargs):
    """@brief Loading a binary-JSON/BJData string buffer and decode it to native Python data

    @param[in] bytes: a BJData byte-buffer or byte-stream
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    opt.setdefault("decode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = False

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        data = bjdata.loadb(bytes, **kwargs)
        if opt["decode"]:
            data = jd.decode(data, opt)
        return data


def show(data, opt={}, **kwargs):
    """@brief Printing a python data as JSON string or return the JSON string (opt['string']=True)

    @param[in] data: data to be saved
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before printing
    """

    kwargs.setdefault("default", jd.jsonfilter)
    opt.setdefault("string", False)
    opt.setdefault("encode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = True

    if opt["encode"]:
        data = jd.encode(data, opt)

    str = json.dumps(data, **kwargs)

    if opt["string"]:
        return str
    else:
        print(str)


def dumpb(data, opt={}, **kwargs):
    """@brief Printing native python data in binary JSON stream

    @param[in] data: data to be saved
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before printing
    """

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        return bjdata.dumpb(data)


##====================================================================================
## Loading and saving binary JData (i.e. UBJSON) files
##====================================================================================


def loadb(fname, opt={}, **kwargs):
    """@brief Loading a binary (BJData/UBJSON) JData file and decode it to native Python data

    @param[in] fname: a binary (BJData/UBJSON based) JData file name
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    """
    opt.setdefault("decode", True)
    opt.setdefault("inplace", True)
    opt["base64"] = False

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        with open(fname, "rb") as fid:
            data = bjdata.load(fid, **kwargs)
        if opt["decode"]:
            data = jd.decode(data, opt)
        return data


def saveb(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a binary JData (BJData/UBJSON) file

    @param[in] data: data to be saved
    @param[in] fname: a binary (BJData/UBJSON based) JData file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    opt.setdefault("encode", True)
    opt.setdefault("inplace", True)

    try:
        import bjdata
    except ImportError:
        raise ImportError(
            'To read/write binary JData files, you must install the bjdata module by "pip install bjdata"'
        )
    else:
        if opt["encode"]:
            data = jd.encode(data, opt)
        with open(fname, "wb") as fid:
            bjdata.dump(data, fid, **kwargs)


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
                suffix = re.search("\.\w{1,5}(?=([#&].*)*$)", link)
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
            spl = os.path.splitext(link)
            ext = spl[1].lower()
            filename = fname + ext
        if p is not None:
            cachepath.pop(1)
        else:
            cachepath.pop(0)
        return cachepath, filename


def jdlink(uripath, opt={}, **kwargs):
    """@brief Printing the local folder and file name where a linked data file in the URL to be saved

    @param[in] url: a URL
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    """

    opt.setdefault("showlink", 1)
    opt.setdefault("showsize", 1)

    if isinstance(uripath, list):
        if "regex" in opt:
            pat = re.compile(opt["regex"])
            uripath = [uri for uri in uripath if pat.search(uri)]
            print(uripath)
        if "showsize" in opt:
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
            newdata, fname, cachepath = downloadlink(uripath[i], opt, **kwargs)
            alloutput[0].append(newdata)
            alloutput[1].append(fname)
            alloutput[2].append(cachepath)
        if len(uripath) == 1:
            alloutput = [x[0] for x in alloutput]
        newdata, fname, cachepath = tuple(alloutput)
    elif isinstance(uripath, str):
        newdata, fname, cachepath = downloadlink(uripath, opt, **kwargs)
    return newdata, fname


def downloadlink(uripath, opt={}, **kwargs):
    opt.setdefault("showlink", 1)

    if "nocache" in opt and opt["nocache"]:
        newdata = urllib.request.urlopen(uripath).read()
        try:
            newdata = loadts(newdata, opt, **kwargs)
        except:
            try:
                newdata = loadbs(newdata, opt, **kwargs)
            except:
                pass
        return newdata, uripath, None

    newdata = []
    cachepath, filename = jsoncache(uripath)
    if isinstance(cachepath, list) and cachepath:
        if opt["showlink"]:
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
        if ext in jext["t"] or ext in jext["b"]:
            newdata = jd.load(fname, opt)

    elif not isinstance(cachepath, list) and os.path.exists(cachepath):
        if opt["showlink"]:
            print("loading from cache:", cachepath)
        fname = cachepath
        spl = os.path.splitext(fname)
        ext = spl[1].lower()
        if ext in jext["t"] or ext in jext["b"]:
            newdata = jd.load(fname, opt)
    return newdata, fname, cachepath
