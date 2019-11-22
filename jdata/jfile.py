"""@package docstring
File IO to load/decode JData-based files to Python data or encode/save Python data to JData files

Copyright (c) 2019 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ['load','save','show','loadt','savet','loadb','saveb','jext']

##====================================================================================
## dependent libraries
##====================================================================================

import json
import os
import jdata as jd

##====================================================================================
## global variables
##====================================================================================

jext={'t':['.json','.jdat','.jnii','.jmsh'], 'b':['.ubj','.jbat','.bnii','.bmsh']};

##====================================================================================
## Loading and saving data based on file extensions
##====================================================================================

def load(fname, opt={}, **kwargs):
    """@brief Loading a JData file (binary or text) according to the file extension
    
    @param[in] fname: a JData file name (accept .json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if(ext in jext['t']):
        return loadt(fname, opt, **kwargs);
    elif(ext in jext['b']):
        return loadb(fname, opt, **kwargs);
    else:
        raise Exception('JData', 'file extension is not recognized, accept (.json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)')

def save(data, fname, opt={}, **kwargs):
    """@brief Saving Python data to file (binary or text) according to the file extension
    
    @param[in] data: data to be saved
    @param[in] fname: a JData file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if(ext in jext['t']):
        savet(data, fname, opt, **kwargs);
    elif(ext in jext['b']):
        saveb(data, fname, opt, **kwargs);
    else:
        raise Exception('JData', 'file extension is not recognized, accept (.json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)')

##====================================================================================
## Loading and saving text-based JData (i.e. JSON) files
##====================================================================================

def loadt(fname, opt={}, **kwargs):
    """@brief Loading a text-based (JSON) JData file and decode it to native Python data
    
    @param[in] fname: a text JData (JSON based) file name
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() after loading
    """
    with open(fname, "r") as fid:
        data=json.load(fid, strict=False, **kwargs);
    if(not ('decode' in opt and not(opt['decode'])) ):
        data=jd.decode(data,opt);
    return data

def savet(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a text-based JData (JSON) file
    
    @param[in] data: data to be saved
    @param[in] fname: a text JData (JSON based) file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    if(not ('encode' in opt and not(opt['encode'])) ):
        data=jd.encode(data,opt);

    with open(fname, "w") as fid:
        if('default' in kwargs):
            json.dump(data, fid, **kwargs);
        else:
            json.dump(data, fid, default=jd.jsonfilter,**kwargs);

def show(data, opt={}, **kwargs):
    """@brief Printing a python data as JSON string or return the JSON string (opt['string']=True)
    
    @param[in] data: data to be saved
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before printing
    """
    if(not ('encode' in opt and not(opt['encode'])) ):
        data=jd.encode(data,opt);

    if('default' in kwargs):
        str=json.dumps(data, **kwargs);
    else:
        str=json.dumps(data, default=jd.jsonfilter, **kwargs);
    if('string' in opt and opt['string']):
        return str;
    else:
        print(str);

##====================================================================================
## Loading and saving binary JData (i.e. UBJSON) files
##====================================================================================

def loadb(fname, opt={}, **kwargs):
    """@brief Loading a binary (UBJSON) JData file and decode it to native Python data

    @param[in] fname: a binary (UBJSON basede) JData file name
    @param[in] opt: options, if opt['decode']=True or 1 (default), call jdata.decode() before saving
    """
    try:
        import ubjson
    except ImportError:
        raise ImportError('To read/write binary JData files, you must install the py-ubjson module by "pip install py-ubjson"')
    else:
        with open(fname, "r") as fid:
            data=ubjson.load(fid,**kwargs);
        if(not ('decode' in opt and not(opt['decode'])) ):
            data=jd.decode(data,opt);
        return data

def saveb(data, fname, opt={}, **kwargs):
    """@brief Saving a Python data structure to a binary JData (UBJSON) file

    @param[in] data: data to be saved
    @param[in] fname: a binary (UBJSON basede) JData file name
    @param[in] opt: options, if opt['encode']=True or 1 (default), call jdata.encode() before saving
    """
    try:
        import ubjson
    except ImportError:
        raise ImportError('To read/write binary JData files, you must install the py-ubjson module by "pip install py-ubjson"')
    else:
        if(not ('encode' in opt and not(opt['encode'])) ):
            data=jd.encode(data,opt);
        with open(fname, "w") as fid:
            ubjson.dump(data, fid,**kwargs);

##====================================================================================
## helper functions
##====================================================================================
