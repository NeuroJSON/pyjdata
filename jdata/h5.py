"""@package docstring
File IO to load/decode HDF5 or SNIRF/JSNIRF files

Copyright (c) 2019-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "loadh5",
    "saveh5",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np


def loadh5(filename, *args, **kwargs):
    """
    Load data in an HDF5 file into a Python dictionary.

    Parameters:
        filename : str
            HDF5 file name
        args : optional
            May contain root path or option dictionary
        kwargs : optional
            Options such as:
                - Order ('creation' | 'alphabet')
                - Regroup (0 | 1)
                - PackHex (0 | 1)
                - ComplexFormat: [realKey, imagKey]
                - Transpose (0 | 1)

    Returns:
        data : dict
            Dictionary containing datasets
        meta : dict
            Dictionary containing attributes
    """
    import h5py

    # Parse args
    path = ""
    opt = {
        "Transpose": 1,
        "StringArray": 0,
        "PackHex": 1,
        "ComplexFormat": ["Real", "Imag"],
        "Regroup": 0,
        "jdata": 0,
    }

    if len(args) == 1 and isinstance(args[0], str):
        path = args[0]
    elif len(args) >= 2:
        path = args[0]
        opt.update(args[1])
    elif len(args) % 2 == 0:
        for k, v in zip(args[::2], args[1::2]):
            opt[k] = v

    opt.update(kwargs)

    def read_attrs(obj):
        return {k: obj.attrs[k] for k in obj.attrs}

    def fix_data(data, attrs):
        if isinstance(data, np.ndarray):
            if opt["Transpose"] and data.ndim > 1:
                data = data.transpose()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, np.ndarray) and data.dtype == np.object:
            try:
                data = np.array(
                    [d.decode("utf-8") if isinstance(d, bytes) else d for d in data]
                )
            except Exception:
                pass
        if isinstance(data, dict):
            fields = data.keys()
            ck = opt["ComplexFormat"]
            if ck[0] in fields and ck[1] in fields:
                data = np.array(data[ck[0]]) + 1j * np.array(data[ck[1]])
        return data

    def visit_group(g, prefix=""):
        d = {}
        m = {}
        for k in g:
            item = g[k]
            name = k
            if isinstance(item, h5py.Group):
                sub_d, sub_m = visit_group(item, prefix + "/" + k)
                d[name] = sub_d
                m[name] = sub_m
            elif isinstance(item, h5py.Dataset):
                try:
                    raw = item[()]
                    attr = read_attrs(item)
                    raw = fix_data(raw, attr)
                    d[name] = raw
                    m[name] = attr
                except Exception as e:
                    d[name] = None
                    m[name] = {"error": str(e)}
        return d, m

    with h5py.File(filename, "r") as f:
        if path and path in f:
            root = f[path]
        else:
            root = f
        data, meta = visit_group(root)

    return data, meta


def saveh5(data, fname, **kwargs):
    """
    Save a Python dictionary or object into an HDF5 file.

    Parameters:
        data : dict, list, or array-like
            Data to be saved.
        fname : str
            Output HDF5 filename.
        kwargs : optional arguments for customization
            Supported keys:
                - rootname (str)
                - compression ('gzip' or None)
                - compresslevel (int)
                - transpose (bool)
                - complex_format (tuple of str)
    """
    import h5py

    rootname = kwargs.get("rootname", "data")
    compression = kwargs.get("compression", None)
    compresslevel = kwargs.get("compresslevel", 4)
    transpose = kwargs.get("transpose", True)
    complex_format = kwargs.get("complex_format", ("Real", "Imag"))

    def write_data(h5file, path, value):
        if isinstance(value, dict):
            grp = h5file.require_group(path)
            for k, v in value.items():
                write_data(h5file, f"{path}/{k}", v)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(i, dict) for i in value
        ):
            for i, v in enumerate(value):
                write_data(h5file, f"{path}/{i}", v)
        elif isinstance(value, complex):
            grp = h5file.require_group(path)
            grp.create_dataset(complex_format[0], data=np.real(value))
            grp.create_dataset(complex_format[1], data=np.imag(value))
        elif isinstance(value, np.ndarray) and np.iscomplexobj(value):
            grp = h5file.require_group(path)
            grp.create_dataset(
                complex_format[0],
                data=np.real(value),
                compression=compression,
                compression_opts=compresslevel,
            )
            grp.create_dataset(
                complex_format[1],
                data=np.imag(value),
                compression=compression,
                compression_opts=compresslevel,
            )
        elif isinstance(value, (np.ndarray, list, tuple)):
            arr = np.array(value)
            if transpose and arr.ndim > 1:
                arr = arr.T
            h5file.create_dataset(
                path, data=arr, compression=compression, compression_opts=compresslevel
            )
        else:
            h5file.create_dataset(path, data=value)

    with h5py.File(fname, "w") as f:
        write_data(f, rootname, data)
