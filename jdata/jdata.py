"""Encoding and decoding python native data structures as 
portable JData-spec annotated dict structure

Copyright (c) 2019 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ['encode','decode']


import numpy as np
import zlib
import base64
import copy

jdtype={'float32':'single','float64':'double','float_':'double',
'bool':'uint8','byte':'int8','short':'int16','ubyte':'uint8',
'ushort':'uint16','int_':'int32','uint':'uint32',
'longlong':'int64','ulonglong':'uint64','csingle':'single','cdouble':'double'};

def encode(d, opt={}):
    """encode a Python data structure."""
    if isinstance(d, str) or isinstance(d, int) or isinstance(d, bool):
        return d;
    elif isinstance(d, float):
	if(np.isnan(d)):
	    return '_NaN_';
	elif(np.isinf(d)):
	    return '_Inf_' if (d>0) else '-_Inf_';
	return d;
    elif isinstance(d, list) or isinstance(d, tuple) or isinstance(d, set) or isinstance(d, frozenset):
	return encodelist(d,opt);
    elif isinstance(d, dict):
	return encodedict(d,opt);
    elif isinstance(d, complex):
	newobj={
	    '_ArrayType_': 'double',
	    '_ArraySize_': 1,
	    '_ArrayIsComplex_': True,
	    '_ArrayData_': [d.real, d.imag]
	  };
    elif isinstance(d, np.ndarray):
	newobj={};
	newobj["_ArrayType_"]=jdtype[str(d.dtype)] if (str(d.dtype) in jdtype) else str(d.dtype);
	newobj["_ArraySize_"]=list(d.shape);
	if(d.dtype==np.complex64 or d.dtype==np.complex128 or d.dtype==np.csingle or d.dtype==np.cdouble):
		newobj['_ArrayIsComplex_']=True;
        	newobj['_ArrayData_']=[list(d.flatten().real), list(d.flatten().imag)];
	else:
        	newobj["_ArrayData_"]=list(d.flatten());
	if('compression' in opt and opt['compression']=='zlib'):
		newobj['_ArrayZipType_']=opt['compression'];
		newobj['_ArrayZipSize_']=[1+int('_ArrayIsComplex_' in newobj), d.size];
		newobj['_ArrayZipData_']=np.asarray(newobj['_ArrayData_'],dtype=d.dtype).tostring();
		newobj['_ArrayZipData_']=base64.b64encode(zlib.compress(newobj['_ArrayZipData_']));
		newobj.pop('_ArrayData_');
    else:
	return copy.deepcopy(d);
    return newobj;


def decode(d, opt={}):
    """encode a Python data structure."""

    if isinstance(d, str) and len(d)<=6 and d[-1]=='_':
	if(d=='_NaN_'):
	    return float('nan');
	elif(d=='_Inf_'):
	    return float('inf');
	elif(d=='-_Inf_'):
	    return float('-inf');
        return d;
    elif isinstance(d, list) or isinstance(d, tuple) or isinstance(d, set) or isinstance(d, frozenset):
	return decodelist(d,opt);
    elif isinstance(d, dict):
	if('_ArrayType_' in d):
	    if('_ArrayZipData_' in d):
		newobj=np.fromstring(base64.b64decode(zlib.uncompress(d['_ArrayZipData_'])),dtype=np.dtype(d['_ArrayType_']));
		newobj=newobj.reshape(d['_ArrayZipSize_']);
		if('_ArrayIsComplex_' in d and newobj.shape[0]==2):
		    newobj=newobj[0]+1j*newobj[1];
		newobj=newobj.reshape(d['_ArraySize_']);
		return newobj;
	    elif('_ArrayData_' in d):
		newobj=np.asarray(d['_ArrayData_'],dtype=np.dtype(d['_ArrayType_']));
		if('_ArrayIsComplex_' in d and newobj.shape[0]==2):
		    newobj=newobj[0]+1j*newobj[1];
		newobj=newobj.reshape(d['_ArraySize_']);
		return newobj;
	return decodedict(d,opt);
    else:
	return copy.deepcopy(d);

def encodedict(d0, opt={}):
    d=dict(d0);
    for k, v in d.items():
	d[k]=encode(v,opt);
    return d;

def encodelist(d, opt={}):
    for i, s in enumerate(d):
        d[i] = encode(s,opt);
    return d;

def decodedict(d0, opt={}):
    d=dict(d0);
    for k, v in d.items():
	d[k]=decode(v,opt);
    return d;

def decodelist(d, opt={}):
    for i, s in enumerate(d):
        d[i] = decode(s,opt);
    return d;
