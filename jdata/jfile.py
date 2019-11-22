"""@package docstring
File IO to load/decode JData-based files to Python data or encode/save Python data to JData files

Copyright (c) 2019 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ['load','save','loadt','savet','loadb','saveb']

##====================================================================================
## dependent libraries
##====================================================================================

import json
import os
import jdata as jd

##====================================================================================
## global variables
##====================================================================================

##====================================================================================
## Loading and saving data based on file extensions
##====================================================================================

def load(fname, opt={}):
    """@brief Loading a JData file (binary or text) according to the file extension
    """
    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if(ext == '.json' or ext=='.jdat' or ext=='.jnii' or ext =='.jmsh'):
	return loadt(fname, opt);
    elif(ext =='.ubj' or ext=='.jbat' or ext=='.bnii' or ext =='.bmsh'):
	return loadb(fname, opt);
    else:
	raise Exception('JData', 'file extension is not recognized, accept (.json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)')

def save(data, fname, opt={}):
    """@brief Saving a JData data to file (binary or text) according to the file extension
    """
    spl = os.path.splitext(fname)
    ext = spl[1].lower()

    if(ext == '.json' or ext=='.jdat' or ext=='.jnii' or ext =='.jmsh'):
	savet(data, fname, opt);
    elif(ext =='.ubj' or ext=='.jbat' or ext=='.bnii' or ext =='.bmsh'):
	saveb(data, fname, opt);
    else:
	raise Exception('JData', 'file extension is not recognized, accept (.json,.jdat,.jbat,.jnii,.bnii,.jmsh,.bmsh)')

##====================================================================================
## Loading and saving text-based JData (i.e. JSON) files
##====================================================================================

def loadt(fname, opt={}):
    """@brief Encoding a Python data structure to portable JData-annotated dict constructs
    
    This function converts complex data types (usually not JSON-serializable) into
    portable JData-annotated dict/list constructs that can be easily exported as JSON/JData
    files
    
    @param[in] fname: a JData file name
    @param[in] opt: options, accepted fields include 'decode'={True,False} to call JData decode after loading
    """
    with open(fname, "r") as fid:
        data=json.load(fid, strict=False);
    if(not ('decode' in opt and not(opt['decode'])) ):
	data=jd.decode(data,opt);
    return data

def savet(data, fname, opt={}):
    """@brief Encoding a Python data structure to portable JData-annotated dict constructs
    
    This function converts complex data types (usually not JSON-serializable) into
    portable JData-annotated dict/list constructs that can be easily exported as JSON/JData
    files

    @param[in] data: data to be saved
    @param[in] fname: a JData file name
    @param[in] opt: options, accepted fields include 'encode'={True,False} to call JData encode before saving
    """
    if(not ('encode' in opt and not(opt['encode'])) ):
	data=jd.encode(data,opt);

    with open(fname, "w") as fid:
        json.dump(data, fid, default=jd.jsonfilter);

##====================================================================================
## Loading and saving binary JData (i.e. UBJSON) files
##====================================================================================

def loadb(fname, opt={}):
    """@brief Encoding a Python data structure to portable JData-annotated dict constructs
    
    This function converts complex data types (usually not JSON-serializable) into
    portable JData-annotated dict/list constructs that can be easily exported as JSON/JData
    files
    
    @param[in] fname: a JData file name
    @param[in] opt: options, accepted fields include 'decode'={True,False} to call JData decode after loading
    """
    import ubjson

    with open(fname, "r") as fid:
        data=ubjson.load(fid);
    if(not ('decode' in opt and not(opt['decode'])) ):
	data=jd.decode(data,opt);
    return data

def saveb(data, fname, opt={}):
    """@brief Encoding a Python data structure to portable JData-annotated dict constructs
    
    This function converts complex data types (usually not JSON-serializable) into
    portable JData-annotated dict/list constructs that can be easily exported as JSON/JData
    files

    @param[in] data: data to be saved
    @param[in] fname: a JData file name
    @param[in] opt: options, accepted fields include 'encode'={True,False} to call JData encode before saving
    """
    import ubjson

    if(not ('encode' in opt and not(opt['encode'])) ):
	data=jd.encode(data,opt);

    with open(fname, "w") as fid:
        ubjson.dump(data, fid);

##====================================================================================
## helper functions
##====================================================================================
