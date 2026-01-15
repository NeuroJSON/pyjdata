"""@package docstring
File IO to load/decode HDF5 or SNIRF/JSNIRF files

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "loadh5",
    "saveh5",
    "regrouph5",
    "aos2soa",
    "soa2aos",
    "jsnirfcreate",
    "snirfcreate",
    "snirfdecode",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import re
import os
from typing import Dict, Any, Tuple, Union, Optional, List
from datetime import datetime
from collections import OrderedDict
import warnings
from .jdata import jdatadecode, jdataencode

##====================================================================================
## implementation
##====================================================================================


def loadh5(filename, *args, **kwargs) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Load data in an HDF5 file to a Python dictionary structure.

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        filename: Name of the file to load data from, or an h5py.File handle
        *args: Optional positional arguments (rootpath, options)
        **kwargs: Optional keyword arguments for user specified options
            Order: 'creation' - creation order (default), or 'alphabet' - alphabetic
            Regroup: [0|1]: if 1, call regrouph5() to combine indexed
                  groups into a cell array
            PackHex: [1|0]: convert invalid characters in the group/dataset
                  names to 0x[hex code] by calling encodevarname;
                  if set to 0, call getvarname
            ComplexFormat: ['realKey','imagKey']: use 'realKey' and 'imagKey'
                  as possible keywords for the real and the imaginary part
                  of a complex array, respectively (sparse arrays not supported);
                  a common list of keypairs is used even without this option
            Transpose: [1|0] - if set to 1 (default), the row-majored HDF5
                  datasets are transposed (to column-major) so that the
                  output array has the same dimensions as in the
                  HDF5 dataset header.

    Returns:
        data: a dictionary structure (array)
        meta: optional output to store the attributes stored in the file

    Example:
        a = [np.random.rand(2,2), {'va': 1, 'vb': 'string'}, 1+2j]
        jd.save(a, 'test.h5')
        a2 = loadh5('test.h5')
        a3 = loadh5('test.h5', regroup=1)
        a4 = loadh5('test.h5', '/a1')

    This function was adapted from h5load.m by Pauli Virtanen <pav at iki.fi>
    This file is part of EasyH5 Toolbox: https://github.com/NeuroJSON/easyh5

    License: GPLv3 or 3-clause BSD license, see https://github.com/NeuroJSON/easyh5 for details
    """
    import h5py

    def load_one(loc: Union[h5py.File, h5py.Group], opt: Dict) -> Tuple[Dict, Dict]:
        """
        Load data from one HDF5 group or file

        Args:
            loc: HDF5 file or group handle
            opt: options dictionary

        Returns:
            data: loaded data dictionary
            meta: metadata dictionary
        """

        data = {}
        meta = {}
        inputdata = {"data": data, "meta": meta, "opt": opt}

        # Load groups and datasets
        try:
            # Get items in order
            if opt.get("order") == "alphabet":
                items = sorted(loc.keys())
            else:
                items = list(loc.keys())  # h5py maintains creation order by default

            for objname in items:
                group_iterate(loc, objname, inputdata)

        except Exception as e:
            raise e

        return inputdata["data"], inputdata["meta"]

    def group_iterate(
        group_id: Union[h5py.File, h5py.Group], objname: str, inputdata: Dict
    ) -> Dict:
        """
        Iterate through HDF5 groups and datasets

        Args:
            group_id: HDF5 group handle
            objname: object name
            inputdata: input data structure

        Returns:
            Updated inputdata structure
        """

        attr = {}

        try:
            data = inputdata["data"]
            meta = inputdata["meta"]

            obj = group_id[objname]

            if isinstance(obj, h5py.Group):
                # Group
                name = re.sub(r".*/", "", objname)

                try:
                    sub_data, sub_meta = load_one(obj, inputdata["opt"])
                except Exception as e:
                    raise e

                data[name] = sub_data
                meta[name] = sub_meta

            elif isinstance(obj, h5py.Dataset):
                # Dataset
                name = re.sub(r".*/", "", objname)

                try:
                    sub_data = obj[()]  # Read entire dataset

                    # Get attributes
                    try:
                        for attr_name in obj.attrs:
                            attr[attr_name] = obj.attrs[attr_name]
                    except Exception:
                        attr = {}

                except Exception as e:
                    raise e

                # Handle string data
                if isinstance(sub_data, bytes):
                    sub_data = sub_data.decode("utf-8")
                elif isinstance(sub_data, np.ndarray) and sub_data.dtype.kind in [
                    "S",
                    "U",
                ]:
                    if sub_data.ndim == 0:
                        sub_data = str(sub_data.item())
                    else:
                        sub_data = sub_data.astype(str)

                # Transpose if requested (for numeric data or multi-element cells)
                if (
                    isinstance(sub_data, np.ndarray)
                    and np.issubdtype(sub_data.dtype, np.number)
                    and inputdata["opt"]["dotranspose"]
                ) or (isinstance(sub_data, (list, tuple)) and len(sub_data) > 1):
                    if isinstance(sub_data, np.ndarray) and sub_data.ndim > 1:
                        # Transpose by reversing dimensions (F-order)
                        sub_data = np.transpose(
                            sub_data, axes=tuple(range(sub_data.ndim - 1, -1, -1))
                        )

                sub_data = fix_data(sub_data, attr, inputdata["opt"])

                data[name] = sub_data
                meta[name] = attr

        except Exception as e:
            raise e

        return {"data": data, "meta": meta, "opt": inputdata["opt"]}

    def fix_data(data: Any, attr: Dict, opt: Dict) -> Any:
        """
        Fix some common types of data to more friendly form.

        Args:
            data: input data
            attr: attributes dictionary
            opt: options dictionary

        Returns:
            Fixed data
        """

        if isinstance(data, dict):
            fields = list(data.keys())

            # Handle sparse arrays
            if len(set(fields) & {"SparseIndex", opt["complexformat"][0]}) == 2:
                if isinstance(data.get("SparseIndex"), np.ndarray) and isinstance(
                    data.get(opt["complexformat"][0]), np.ndarray
                ):
                    if attr and "SparseArraySize" in attr:
                        from scipy.sparse import csr_matrix

                        size = attr["SparseArraySize"]
                        total_size = np.prod(size)
                        spd = np.zeros(
                            total_size,
                            dtype=complex if opt["complexformat"][1] in data else float,
                        )

                        # Convert 1-based indices to 0-based
                        indices = data["SparseIndex"] - 1

                        if opt["complexformat"][1] in data:
                            spd[indices] = (
                                data[opt["complexformat"][0]]
                                + 1j * data[opt["complexformat"][1]]
                            )
                        else:
                            spd[indices] = data[opt["complexformat"][0]]

                        data = spd.reshape(size, order="F")
                        return data
            else:
                # Handle complex numbers
                complex_pairs = [
                    opt["complexformat"],
                    ["Real", "Imag"],
                    ["real", "imag"],
                    ["Re", "Im"],
                    ["re", "im"],
                    ["r", "i"],
                ]

                for real_key, imag_key in complex_pairs:
                    if len(set(fields) & {real_key, imag_key}) == 2:
                        if isinstance(
                            data.get(real_key), (np.ndarray, int, float)
                        ) and isinstance(data.get(imag_key), (np.ndarray, int, float)):
                            data = data[real_key] + 1j * data[imag_key]
                            break

        # Handle string arrays
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            if isinstance(data, np.ndarray) and data.dtype.kind in ["S", "U"]:
                data = data.astype(str)
            elif isinstance(data, list) and all(
                isinstance(x, (str, bytes)) for x in data
            ):
                if opt.get("stringarray", 0):
                    data = np.array(data, dtype=str)

        return data

    path = ""
    opt = {}

    # Process arguments similar to MATLAB's varargin
    if len(args) % 2 == 0:
        # Even number of args - treat as key-value pairs
        for i in range(0, len(args), 2):
            opt[args[i]] = args[i + 1]
    elif len(args) >= 3:
        # First arg is path, rest are key-value pairs
        path = args[0]
        for i in range(1, len(args), 2):
            if i + 1 < len(args):
                opt[args[i]] = args[i + 1]
    elif len(args) == 1:
        path = args[0]

    # Set default options
    opt["dotranspose"] = opt.get("Transpose", opt.get("transpose", 0))
    opt["stringarray"] = opt.get("StringArray", opt.get("stringarray", 0))
    opt["rootpath"] = path

    # Merge kwargs into opt
    opt.update(kwargs)

    # Handle file opening
    if isinstance(filename, h5py.File):
        loc = filename
        close_file = False
    else:
        try:
            loc = h5py.File(filename, "r")
            close_file = True
        except Exception:
            raise ValueError("fail to open file")

    # Set complex format
    if not (
        "complexformat" in opt
        and isinstance(opt["complexformat"], list)
        and len(opt["complexformat"]) == 2
    ):
        opt["complexformat"] = ["Real", "Imag"]

    # Set ordering preference
    if opt.get("order", "").lower() == "alphabet":
        opt["order"] = "alphabet"
    else:
        opt["order"] = "creation"

    try:
        if path:
            try:
                # Try to open as group first
                if path in loc:
                    rootgid = loc[path]
                    data, meta = load_one(rootgid, opt)
                else:
                    # Try as dataset
                    gname = "/".join(path.split("/")[:-1]) or "/"
                    dname = path.split("/")[-1]
                    if gname in loc:
                        rootgid = loc[gname]
                        data, meta = {}, {}
                        group_iterate(
                            rootgid, dname, {"data": data, "meta": meta, "opt": opt}
                        )
                    else:
                        raise KeyError(f"Path {path} not found")
            except Exception:
                raise ValueError(f"Cannot access path {path}")
        else:
            data, meta = load_one(loc, opt)

        if close_file:
            loc.close()
    except Exception as e:
        if close_file:
            loc.close()
        raise e

    # Apply regrouping if requested
    if opt.get("regroup", 0):
        data = regrouph5(data)

    # Apply jdata decoding if requested
    if opt.get("jdata", 0):
        data = jdatadecode(data, base64=False, **opt)

    # Return based on number of requested outputs
    return data, meta


def saveh5(data: Any, fname, *args, **kwargs):
    """
    Save a Python dict (array) or list (array) into an HDF5 file

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        data: a dictionary structure (array) or list (array) to be stored.
        fname: the output HDF5 (.h5) file name or h5py.File handle
        *args: optional positional arguments
        **kwargs: optional keyword arguments for user specified options
            JData [0|1] use JData Specification to serialize complex data structures
                         such as complex/sparse arrays, tables, maps, graphs etc by
                         calling jdataencode before saving data to HDF5
            RootName: the HDF5 path of the root object. If not given, the
                         actual variable name for the data input will be used as
                         the root object. The value shall not include '/'.
            UnpackHex [1|0]: convert the 0x[hex code] in variable names
                         back to Unicode string using decodevarname
            Compression: ['gzip'|''] - use gzip method
                         to compress data array
            CompressArraySize: [100|int]: only to compress an array if the
                         total element count is larger than this number.
            CompressLevel: [5|int] - a number between 1-9 to set
                         compression level
            Chunk: a size vector or None - breaking a large array into
                         small chunks of size specified by this parameter
            Append [0|1]: if set to 1, new data will be appended to a
                         file if already exists under the user-defined
                         'rootname' path; if set to 0, old data
                         will be overwritten if the file exists.
            VariableLengthString [0|1]: if set to 1, strings and char arrays will be
                         saved with variable length string type
            Scalar [1|0]: if set to 1, arrays of length 1 will be saved as
                         a scalar instead of a length-1 array
            Transpose: [1|0] - if set to 1 (default), arrays are
                         transposed (from column-major to row-major) so
                         that the output HDF5 dataset shows the same
                         dimensions as in Python when reading from other
                         tools.
            ComplexFormat: ['realKey','imagKey']: use 'realKey' and 'imagKey'
                  as keywords for the real and the imaginary part of a
                  complex array, respectively (sparse arrays not supported);
                  the default values are ['Real','Imag']

    Example:
        a = {'a': np.random.rand(5,5), 'b': 'string', 'c': True, 'd': 2+3j, 'e': ['test', None, list(range(1,6))]}
        saveh5(a, 'test.h5')
        saveh5(a, 'test2.h5', rootname='')
        saveh5(a, 'test2.h5', compression='gzip', compressarraysize=1)
        saveh5(a, 'test.h5j', jdata=1)
        saveh5(a, 'test.h5j', rootname='/newroot', append=1)

    This file is part of EasyH5 Toolbox: https://github.com/NeuroJSON/easyh5

    License: GPLv3 or 3-clause BSD license, see https://github.com/NeuroJSON/easyh5 for details
    """
    import h5py

    def obj2h5(
        name: str, item: Any, handle: h5py.Group, level: int, opt: Dict
    ) -> Optional[h5py.Dataset]:
        """
        Convert Python object to HDF5 dataset/group

        Args:
            name: HDF5 path name
            item: Python object to save
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            HDF5 dataset/group handle or None
        """

        if isinstance(item, list):
            return cell2h5(name, item, handle, level, opt)
        elif isinstance(item, dict):
            return struct2h5(name, item, handle, level, opt)
        elif isinstance(item, (str, bytes)):
            return mat2h5(name, item, handle, level, opt)
        elif isinstance(item, (bool, int, float, complex, np.ndarray)):
            return mat2h5(name, item, handle, level, opt)
        else:
            return OrderedDict()

    def idxobj2h5(
        name: str, idx: int, item: Any, handle: h5py.Group, level: int, opt: Dict
    ) -> Optional[h5py.Dataset]:
        """
        Save indexed object to HDF5

        Args:
            name: base name
            idx: index number
            item: object to save
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            HDF5 dataset/group handle
        """
        return obj2h5(f"{name}{idx}", item, handle, level, opt)

    def cell2h5(
        name: str, item: List, handle: h5py.Group, level: int, opt: Dict
    ) -> List:
        """
        Convert Python list to HDF5 groups/datasets

        Args:
            name: HDF5 path name
            item: list to save
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            List of HDF5 handles
        """

        num = len(item)
        if num > 1:
            # Create indexed datasets for multi-element lists
            oid = []
            for i, x in enumerate(item, 1):  # Start from 1 for MATLAB compatibility
                oid.append(idxobj2h5(name, i, x, handle, level, opt))
            return oid
        else:
            # Single element
            return [obj2h5(name, item[0], handle, level, opt)] if item else []

    def struct2h5(
        name: str, item: Dict, handle: h5py.Group, level: int, opt: Dict
    ) -> List:
        """
        Convert Python dictionary to HDF5 group

        Args:
            name: HDF5 path name
            item: dictionary to save
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            List of HDF5 handles
        """

        try:
            # Create new group
            group_handle = handle.create_group(name)
            isnew = True
        except ValueError:
            # Group already exists
            group_handle = handle[name]
            isnew = False

        names = list(item.keys())
        oid = []

        for field_name in names:
            field_handle = obj2h5(
                field_name, item[field_name], group_handle, level + 1, opt
            )
            oid.append(field_handle)

        return oid

    def mat2h5(
        name: str, item: Any, handle: h5py.Group, level: int, opt: Dict
    ) -> Optional[h5py.Dataset]:
        """
        Convert numeric/string data to HDF5 dataset

        Args:
            name: HDF5 path name
            item: data to save
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            HDF5 dataset handle
        """
        from scipy.sparse import issparse

        is1dvector = False

        # Convert to numpy array if needed
        if not isinstance(item, np.ndarray):
            if isinstance(item, (list, tuple)):
                item = np.array(item, order="F")
            elif isinstance(item, (int, float, complex, bool)):
                item = np.array(item)
            elif isinstance(item, str):
                item = np.array(item.encode("utf-8") if isinstance(item, str) else item)

        # Handle transpose
        if opt["dotranspose"] and isinstance(item, np.ndarray) and item.ndim > 1:
            # Transpose by reversing dimensions for F-order
            item = np.transpose(item, axes=tuple(range(item.ndim - 1, -1, -1)))

        # Set up compression
        compression = None
        compression_opts = None
        chunks = None

        if (
            opt["compression"]
            and isinstance(item, np.ndarray)
            and item.size >= opt["compressarraysize"]
        ):
            if opt["compression"].lower() in ["deflate", "gzip"]:
                compression = "gzip"
                compression_opts = opt["compresslevel"]
                chunks = True  # Enable chunking for compression
                opt["scalar"] = False
                opt["variablelengthstring"] = False

        # Handle empty arrays
        if isinstance(item, np.ndarray) and item.size == 0 and opt["skipempty"]:
            warnings.warn(f'Skip saving empty dataset "{name}"')
            return None

        # Handle complex numbers
        if isinstance(item, np.ndarray) and np.iscomplexobj(item):
            if issparse(item):
                # Handle complex sparse arrays
                idx = find(item)
                sparse_data = {
                    "Size": item.shape,
                    "SparseIndex": idx[0] * item.shape[1]
                    + idx[1]
                    + 1,  # Convert to 1-based linear index
                    "Real": item.data.real,
                    "Imag": item.data.imag,
                }
                return sparse2h5(name, sparse_data, handle, level, opt)
            else:
                # Handle complex dense arrays
                if not (
                    "complexformat" in opt
                    and isinstance(opt["complexformat"], list)
                    and len(opt["complexformat"]) == 2
                ):
                    opt["complexformat"] = ["Real", "Imag"]

                # Create compound datatype for complex numbers
                complex_data = np.empty(
                    item.shape,
                    dtype=[
                        (opt["complexformat"][0], item.real.dtype),
                        (opt["complexformat"][1], item.imag.dtype),
                    ],
                )
                complex_data[opt["complexformat"][0]] = item.real
                complex_data[opt["complexformat"][1]] = item.imag
                item = complex_data

        # Handle real arrays including sparse
        elif isinstance(item, np.ndarray) and issparse(item):
            # Handle real sparse arrays
            idx = find(item)
            sparse_data = {
                "Size": item.shape,
                "SparseIndex": idx[0] * item.shape[1]
                + idx[1]
                + 1,  # Convert to 1-based linear index
                "Real": item.data,
            }
            return sparse2h5(name, sparse_data, handle, level, opt)

        # Determine dataset shape
        if isinstance(item, np.ndarray):
            if item.ndim == 0 or (item.size == 1 and opt["scalar"] and not is1dvector):
                # Scalar
                shape = None
            else:
                shape = item.shape
        else:
            shape = None

        try:
            # Create dataset
            if name in handle:
                if opt.get("append", 0):
                    del handle[name]
                else:
                    raise ValueError(f"Dataset {name} already exists")

            # Handle variable length strings
            if isinstance(item, (str, bytes)) and opt["variablelengthstring"]:
                dt = h5py.string_dtype(encoding="utf-8")
                ds = handle.create_dataset(
                    name,
                    shape,
                    dtype=dt,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks,
                )
            else:
                ds = handle.create_dataset(
                    name,
                    data=item,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks,
                )

            return ds

        except Exception as e:
            if opt.get("append", 0):
                # Try to delete and recreate
                try:
                    del handle[name]
                    ds = handle.create_dataset(
                        name,
                        data=item,
                        compression=compression,
                        compression_opts=compression_opts,
                        chunks=chunks,
                    )
                    return ds
                except Exception:
                    pass
            raise e

    def sparse2h5(
        name: str, item: Dict, handle: h5py.Group, level: int, opt: Dict
    ) -> Optional[h5py.Dataset]:
        """
        Save sparse array data to HDF5

        Args:
            name: HDF5 path name
            item: sparse array data dictionary
            handle: HDF5 group handle
            level: nesting level
            opt: options dictionary

        Returns:
            HDF5 dataset handle
        """

        idx = item["SparseIndex"]

        if len(idx) == 0 and opt["skipempty"]:
            warnings.warn(f'Skip saving empty sparse dataset "{name}"')
            return None

        adata = item["Size"]
        item_copy = item.copy()
        del item_copy["Size"]
        hasimag = "Imag" in item_copy

        # Set up compression
        compression = None
        compression_opts = None
        chunks = None

        if opt["compression"] and len(idx) >= opt["compressarraysize"]:
            if opt["compression"].lower() in ["deflate", "gzip"]:
                compression = "gzip"
                compression_opts = opt["compresslevel"]
                chunks = True

        # Create compound datatype for sparse data
        if hasimag:
            dt = np.dtype(
                [
                    ("SparseIndex", idx.dtype),
                    ("Real", item_copy["Real"].dtype),
                    ("Imag", item_copy["Imag"].dtype),
                ]
            )
            data = np.empty(len(idx), dtype=dt)
            data["SparseIndex"] = idx
            data["Real"] = item_copy["Real"]
            data["Imag"] = item_copy["Imag"]
        else:
            dt = np.dtype(
                [("SparseIndex", idx.dtype), ("Real", item_copy["Real"].dtype)]
            )
            data = np.empty(len(idx), dtype=dt)
            data["SparseIndex"] = idx
            data["Real"] = item_copy["Real"]

        # Create dataset
        ds = handle.create_dataset(
            name,
            data=data,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )

        # Add size attribute
        ds.attrs["SparseArraySize"] = adata

        return ds

    if not data or not fname:
        raise ValueError("you must provide at least two inputs")

    # Default root name
    rootname = "/data"
    opt = {}

    # Process arguments
    if len(args) == 1 and isinstance(args[0], str):
        rootname = args[0] + "/data"
    else:
        # Process key-value pairs from args
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                opt[args[i]] = args[i + 1]

    # Merge kwargs
    opt.update(kwargs)

    # Set default options
    opt["compression"] = opt.get("Compression", opt.get("compression", ""))
    opt["compresslevel"] = opt.get("CompressLevel", opt.get("compresslevel", 5))
    opt["compressarraysize"] = opt.get(
        "CompressArraySize", opt.get("compressarraysize", 100)
    )
    opt["unpackhex"] = opt.get("UnpackHex", opt.get("unpackhex", 1))
    opt["dotranspose"] = opt.get("Transpose", opt.get("transpose", 0))
    opt["variablelengthstring"] = opt.get(
        "VariableLengthString", opt.get("variablelengthstring", 0)
    )
    opt["scalar"] = opt.get("Scalar", opt.get("scalar", 1))
    opt["skipempty"] = False  # h5py handles empty datasets

    # Handle root name
    if "rootname" in opt:
        rootname = "/" + opt["rootname"]

    if rootname.endswith("/"):
        rootname = rootname + "data"

    # Apply JData encoding if requested
    if opt.get("JData", opt.get("jdata", 0)):
        data = jdataencode(data, base64=False, **opt)

    try:
        if isinstance(fname, h5py.File):
            fid = fname
            close_file = False
        else:
            if opt.get("append", 0) and os.path.exists(fname):
                fid = h5py.File(fname, "r+")
            else:
                fid = h5py.File(fname, "w")
            close_file = True

        obj2h5(rootname, data, fid, 1, opt)

        if close_file:
            fid.close()

    except Exception as e:
        if "fid" in locals() and close_file:
            fid.close()
        raise e


def regrouph5(root: Union[Dict, Any], *args) -> Dict:
    """
    Processing a loadh5 restored data and merge "indexed datasets", whose
    names start with an ASCII string followed by a contiguous integer
    sequence number starting from 1, into a cell array. For example,
    datasets {data.a1, data.a2, data.a3} will be merged into a cell/struct
    array data.a with 3 elements.

    A single subfield .name1 will be renamed as .name. Items with
    non-contiguous numbering will not be grouped. If .name and
    .name1/.name2 co-exist in the input struct, no grouping will be done.

    The grouped subfield will appear at the position of the first
    pre-grouped item in the original input structure.

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        root: the raw input HDF5 data structure (loaded from loadh5)
        *args: if args[0] is set as a list of strings, it restricts the
               grouping only to the subset of field names in this list;
               if args[0] is a string as 'snirf', it is the same as setting
               args[0] as ['aux','data','nirs','stim','measurementList'].

    Returns:
        data: a reorganized python dictionary structure.

    Example:
        a = {'a1': np.random.rand(5,5), 'a2': 'string', 'a3': True, 'd': 2+3j, 'e': ['test', None, list(range(1,6))]}
        jd.regrouph5(a)
        jd.saveh5(a, 'test.h5')
        rawdata = jd.loadh5('test.h5')
        data = jd.regrouph5(rawdata)

    This file is part of EasyH5 Toolbox: https://github.com/NeuroJSON/easyh5

    License: GPLv3 or 3-clause BSD license, see https://github.com/NeuroJSON/easyh5 for details
    """

    if root is None:
        return {}

    dict_filter = []
    if len(args) > 0:
        if isinstance(args[0], str) and args[0].lower() == "snirf":
            dict_filter = ["aux", "data", "nirs", "stim", "measurementList"]
        elif isinstance(args[0], (list, tuple)):
            dict_filter = list(args[0])

    data = OrderedDict()

    if isinstance(root, dict):
        # Handle case where root is a single dictionary
        if not isinstance(root, list):
            root_list = [root]
        else:
            root_list = root

        # Initialize data as list of dictionaries if root was a list
        if isinstance(root, list):
            data = [OrderedDict() for _ in range(len(root))]
        else:
            data = [OrderedDict()]

        names = list(root_list[0].keys())
        newnames = {}
        firstpos = {}

        for i, name in enumerate(names):
            # Use regex to match pattern: prefix + digits
            match = re.match(r"^(.*\D)(\d+)$", name)
            if (
                match
                and int(match.group(2)) != 0
                and match.group(1) not in root_list[0]
            ):
                prefix = match.group(1)
                number = int(match.group(2))

                if prefix not in newnames:
                    newnames[prefix] = [number]
                else:
                    newnames[prefix].append(number)

                if prefix not in firstpos:
                    firstpos[prefix] = len(data[0])
            else:
                # Process non-indexed fields
                for j in range(len(root_list)):
                    if isinstance(root_list[j][name], dict):
                        data[j][name] = regrouph5(root_list[j][name])
                    else:
                        data[j][name] = root_list[j][name]

        # Filter names if dictionary filter is provided
        group_names = list(newnames.keys())
        if dict_filter:
            group_names = [name for name in group_names if name in dict_filter]

        # Process grouped fields in reverse order to maintain field positions
        for i in range(len(group_names) - 1, -1, -1):
            name = group_names[i]
            length = len(newnames[name])
            idx = newnames[name]

            # Check if indices are contiguous starting from 1
            if (min(idx) != 1 or max(idx) != length) and length != 1:
                # Non-contiguous indices - keep original field names
                for j in range(length):
                    dataname = f"{name}{idx[j]}"
                    for k in range(len(root_list)):
                        if isinstance(root_list[k][dataname], dict):
                            data[k][dataname] = regrouph5(root_list[k][dataname])
                        else:
                            data[k][dataname] = root_list[k][dataname]

                # Reorder fields to maintain position
                pos = firstpos[name]
                _reorder_dict_fields(data, pos, len(data[0]) - 1)
                continue

            # Create cell array for contiguous indices
            for j in range(len(data)):
                data[j][name] = [None] * length

            idx_sorted = sorted(idx)
            for j in range(length):
                for k in range(len(root_list)):
                    obj = root_list[k][f"{name}{idx_sorted[j]}"]
                    if isinstance(obj, dict):
                        data[k][name][j] = regrouph5(obj)
                    else:
                        data[k][name][j] = obj

            # Reorder fields to maintain position
            pos = firstpos[name]
            _reorder_dict_fields(data, pos, len(data[0]) - 1)

            # Try to convert cell array to matrix if possible
            try:
                # Check if all elements can be converted to numpy array
                first_data = data[0][name]
                if all(
                    isinstance(item, (int, float, complex, np.ndarray))
                    for item in first_data
                ):
                    # Try to stack arrays
                    if all(isinstance(item, np.ndarray) for item in first_data):
                        if all(
                            item.shape == first_data[0].shape for item in first_data
                        ):
                            for j in range(len(data)):
                                # Use Fortran order (column-major) for flattening/stacking
                                data[j][name] = np.stack(data[j][name], axis=0)
                    else:
                        # Convert to numpy array for numeric data
                        for j in range(len(data)):
                            data[j][name] = np.array(data[j][name], order="F")
            except:
                # Keep as list if conversion fails
                pass

    # Return single dictionary if input was single dictionary
    if not isinstance(root, list) and len(data) == 1:
        return OrderedDict(data[0])
    elif isinstance(root, list):
        return [OrderedDict(d) for d in data]
    else:
        return OrderedDict(data[0]) if data else {}


def _reorder_dict_fields(data_list: List[OrderedDict], pos: int, last_pos: int):
    """
    Helper function to reorder dictionary fields to maintain field positions.
    Moves the last field to the specified position.

    Args:
        data_list: List of OrderedDict objects to reorder
        pos: Target position for the last field
        last_pos: Index of the last field
    """
    for data_dict in data_list:
        # Get all items as a list
        items = list(data_dict.items())

        # Move the last item to the target position
        if last_pos < len(items):
            last_item = items.pop(last_pos)
            items.insert(pos, last_item)

            # Rebuild the OrderedDict
            data_dict.clear()
            for key, value in items:
                data_dict[key] = value


def aos2soa(starray: Union[List[Dict], Dict]) -> Dict:
    """
    Convert an array-of-structs (AoS) to a struct-of-arrays (SoA)

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        starray: a list of dictionaries (struct array), with each subfield a simple scalar

    Returns:
        st: a dictionary, containing the same number of subfields as starray
            with each subfield a horizontal-concatenation of the struct
            array subfield values.

    Example:
        a = {'a': 1, 'b': '0', 'c': np.array([1, 3])}
        st = aos2soa([a] * 10)

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if starray is None or not isinstance(starray, (list, dict)):
        raise ValueError("you must give an array of struct")

    # Handle single dictionary case
    if isinstance(starray, dict) or len(starray) <= 1:
        return starray if isinstance(starray, dict) else (starray[0] if starray else {})

    st = {}
    fn = list(starray[0].keys())

    for i in range(len(fn)):
        field_name = fn[i]
        # Collect all values for this field: equivalent to [starray(:).(fn{i})]
        field_values = [struct_item.get(field_name) for struct_item in starray]

        try:
            # Try numpy concatenation with Fortran order for arrays
            if all(isinstance(val, np.ndarray) for val in field_values):
                st[field_name] = np.concatenate(field_values, axis=0)
            else:
                # For scalars and other types, create numpy array or keep as list
                st[field_name] = np.array(field_values, order="F")
        except:
            # Keep as list if concatenation fails
            st[field_name] = field_values

    return st


def soa2aos(starray: Dict) -> List[Dict]:
    """
    Convert a struct-of-arrays (SoA) to an array-of-structs (AoS)

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        starray: a dictionary, with each subfield of numeric vectors

    Returns:
        as: a list of dictionaries, containing the same number of subfields as starray
            with each subfield a single scalar

    Example:
        a = {'a': [1, 2], 'b': [3, 4]}
        st = soa2aos(a)

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if not isinstance(starray, dict):
        raise ValueError("you must give a struct with subfield of numeric vectors")

    # Get sizes of all fields
    allsize = [
        len(v) if hasattr(v, "__len__") and not isinstance(v, str) else 1
        for v in starray.values()
    ]

    # If not all fields have the same size, return original
    if len(set(allsize)) > 1 or not allsize:
        return [starray] if allsize and allsize[0] == 1 else []

    size = allsize[0]
    if size <= 1:
        return [starray] if size == 1 else []

    # Vectorized conversion using zip and dictionary comprehension
    return [
        {
            k: (v[i] if hasattr(v, "__getitem__") and not isinstance(v, str) else v)
            for k, v in starray.items()
        }
        for i in range(size)
    ]


def jsnirfcreate(*args, **kwargs) -> Dict:
    """
    Create an empty JSNIRF data structure defined in the JSNIRF
    specification: https://github.com/NeuroJSON/jsnirf or a SNIRF data structure
    based on the SNIRF specification at https://github.com/fNIRS/snirf

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        *args: if args[0] is a string with value 'snirf', this creates a default
               SNIRF data structure; otherwise, a JSNIRF data structure is created.
        **kwargs: name/value pairs specify additional subfields to be stored
                  under the /nirs object. 'format' can be used instead of args[0].

    Returns:
        jsn: a default SNIRF or JSNIRF data structure.

    Example:
        jsn = jsnirfcreate(data=mydata, aux=myauxdata, comment='test')

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    # Define empty SNIRF data structure with all required fields
    now = datetime.now()

    defaultmeta = OrderedDict(
        [
            ("SubjectID", "default"),
            ("MeasurementDate", now.strftime("%Y-%m-%d")),
            ("MeasurementTime", now.strftime("%H:%M:%S")),
            ("LengthUnit", "mm"),
            ("TimeUnit", "s"),
            ("FrequencyUnit", "Hz"),
        ]
    )

    defaultsrcmap = OrderedDict(
        [
            ("sourceIndex", np.array([], dtype=int)),
            ("detectorIndex", np.array([], dtype=int)),
            ("wavelengthIndex", np.array([], dtype=int)),
            ("dataType", 1),
            ("dataTypeIndex", 1),
        ]
    )

    defaultdata = {
        "dataTimeSeries": np.array([]),
        "time": np.array([]),
        "measurementList": defaultsrcmap,
    }

    defaultaux = {
        "name": "",
        "dataTimeSeries": np.array([]),
        "time": np.array([]),
        "timeOffset": 0,
    }

    defaultstim = {"name": "", "data": np.array([])}

    defaultprobe = {
        "wavelengths": np.array([]),
        "sourcePos2D": np.array([]),
        "detectorPos2D": np.array([]),
    }

    nirsdata = {
        "metaDataTags": defaultmeta,
        "data": defaultdata,
        "aux": defaultaux,
        "stim": defaultstim,
        "probe": defaultprobe,
    }

    # Read user specified data fields
    for key, value in kwargs.items():
        if key.lower() == "format":
            key = "format"
        nirsdata[key] = value

    jsn = OrderedDict()

    # Return either a SNIRF data structure, or JSNIRF data (enclosed in SNIRFData tag)
    is_snirf = (
        len(args) == 1 and isinstance(args[0], str) and args[0].lower() == "snirf"
    ) or (
        "format" in nirsdata
        and isinstance(nirsdata["format"], str)
        and nirsdata["format"].lower() == "snirf"
    )

    if is_snirf:
        if "format" in nirsdata:
            del nirsdata["format"]
        jsn = {"formatVersion": "1.0", "nirs": nirsdata}
    else:
        nirsdata["formatVersion"] = "1.0"
        # Move formatVersion to first position (equivalent to orderfields)
        nirsdata = _move_field_to_first(nirsdata, "formatVersion")
        jsn = {"SNIRFData": nirsdata}

    return jsn


def snirfcreate(*args, **kwargs) -> Dict:
    """
    Create a empty SNIRF data structure defined in the SNIRF
    specification: https://github.com/fNIRS/snirf

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        *args: optional arguments passed to jsnirfcreate
        **kwargs: name/value pairs specify additional subfields to be stored
                  under the /nirs object. 'format' parameter same as option.

    Returns:
        snf: a default SNIRF or JSNIRF data structure.

    Example:
        snf = snirfcreate(data=mydata, aux=myauxdata, comment='test')

    This file is part of JSNIRFY toolbox: https://github.com/NeuroJSON/jsnirfy

    License: GPLv3 or Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if len(args) == 1:
        snf = jsnirfcreate(*args, **kwargs)
    else:
        snf = jsnirfcreate("snirf", *args, **kwargs)

    return snf


def snirfdecode(root: Dict, *args) -> Dict:
    """
    Processing an HDF5 based SNIRF data and group indexed datasets into a
    cell array

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        root: the raw input snirf data structure (loaded from loadh5)
        *args: if args[0] is set as a list of strings, it restricts the
               grouping only to the subset of field names in this list;
               if args[0] is a string as 'snirf', it is the same as setting
               args[0] as ['aux','data','nirs','stim','measurementList'].

    Returns:
        data: a reorganized python dictionary structure. Each SNIRF data chunk is
              enclosed inside a 'SNIRFData' subfield or cell array.

    Example:
        rawdata = jd.load('mydata.snirf', stringarray=True)
        data = snirfdecode(rawdata)

    This file is part of JSNIRF specification: https://github.com/NeuroJSON/jsnirf

    License: Apache 2.0, see https://github.com/NeuroJSON/jsnirfy for details
    """

    if root is None:
        return {}

    data = regrouph5(root, *args)

    issnirf = 1

    if len(args) > 0:
        if isinstance(args[0], str) and args[0].lower() == "jsnirf":
            issnirf = 0

    if (
        issnirf == 0
        and "nirs" in data
        and "formatVersion" in data
        and "SNIRFData" not in data
    ):
        data["SNIRFData"] = data["nirs"]

        # Handle measurementList conversion
        if (
            isinstance(data["SNIRFData"], dict)
            and "data" in data["SNIRFData"]
            and "measurementList" in data["SNIRFData"]["data"]
        ):
            data["SNIRFData"]["data"]["measurementList"] = aos2soa(
                data["nirs"]["data"]["measurementList"]
            )

        # Handle cell array case
        if isinstance(data["nirs"], list):
            for i in range(len(data["nirs"])):
                data["SNIRFData"][i]["formatVersion"] = data["formatVersion"]
                # Move formatVersion to first position (equivalent to orderfields)
                data["SNIRFData"][i] = _move_field_to_first(
                    data["SNIRFData"][i], "formatVersion"
                )
        else:
            data["SNIRFData"]["formatVersion"] = data["formatVersion"]
            # Move formatVersion to first position
            data["SNIRFData"] = _move_field_to_first(data["SNIRFData"], "formatVersion")

        # Remove original fields
        del data["nirs"]
        del data["formatVersion"]

    return data


def _move_field_to_first(data_dict: Dict, field_name: str) -> Dict:
    """
    Helper function to move a field to the first position in dictionary.
    Equivalent to MATLAB's orderfields with [len, 1:len-1] indexing.

    Args:
        data_dict: Dictionary to reorder
        field_name: Field name to move to first position

    Returns:
        Reordered dictionary with field_name as first key
    """
    if not isinstance(data_dict, dict) or field_name not in data_dict:
        return data_dict

    # Create new ordered dictionary with field_name first
    reordered = OrderedDict()
    reordered[field_name] = data_dict[field_name]

    # Add remaining fields
    for key, value in data_dict.items():
        if key != field_name:
            reordered[key] = value

    return dict(reordered)
