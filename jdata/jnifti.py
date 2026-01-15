"""@package docstring
File IO to load/decode NIFTI or JSON-based JNIFTI files

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "nii2jnii",
    "jnii2nii",
    "loadnifti",
    "loadjnifti",
    "savenifti",
    "savejnifti",
    "nifticreate",
    "jnifticreate",
    "memmapstream",
    "niiheader2jnii",
    "niicodemap",
    "niiformat",
    "savejnii",
    "savebnii",
]

##====================================================================================
## dependent libraries
##====================================================================================

import re
import struct
import zlib
import mmap
import warnings
import sys

import numpy as np
import jdata as jd
from typing import Union
from collections import defaultdict


def nii2jnii(filename, format="jnii", *varargin, **kwargs):
    hdrfile = filename
    isnii = -1
    if re.search(
        r"(\.[Hh][Dd][Rr](\.[Gg][Zz])*$|\.[Ii][Mm][Gg](\.[Gg][Zz])*$)", filename
    ):
        isnii = 0
    elif re.search(r"\.[Nn][Ii][Ii](\.[Gg][Zz])*$", filename):
        isnii = 1

    if isnii < 0:
        raise ValueError(
            "file must be a NIfTI (.nii/.nii.gz) or Analyze 7.5 (.hdr/.img,.hdr.gz/.img.gz) data file"
        )

    if re.search(r"\.[Ii][Mm][Gg](\.[Gg][Zz])*$", filename):
        hdrfile = re.sub(r"\.[Ii][Mm][Gg](\.[Gg][Zz])*$", ".hdr\g<1>", filename)

    niftiheader = niiformat("nifti1")

    if re.search(r"\.[Gg][Zz]$", hdrfile):
        with open(hdrfile, "rb") as finput:
            input = finput.read()

        if re.search(r"\.[Gg][Zz]$", hdrfile):
            gzdata = zlib.decompress(bytes(input), zlib.MAX_WBITS | 32)
        else:
            gzdata = input
        nii = {"hdr": memmapstream(gzdata, niftiheader)}
    else:
        import os

        fileinfo = os.stat(hdrfile)
        if fileinfo.st_size == 0:
            raise ValueError("specified file does not exist")
        header = mmap.mmap(
            hdrfile,
            0,
            access=mmap.ACCESS_READ,
            format=niftiheader[: (fileinfo.st_size < 352)],
        )
        nii = {"hdr": header[0]}

    dataendian = sys.byteorder

    if nii["hdr"]["sizeof_hdr"] not in [348, 540]:
        value = nii["hdr"]["sizeof_hdr"]
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        nii["hdr"]["sizeof_hdr"] = value.byteswap()

    if nii["hdr"]["sizeof_hdr"] == 540:  # NIFTI-2 format
        niftiheader = niiformat("nifti2")
        if "gzdata" in locals():
            nii["hdr"] = memmapstream(gzdata, niftiheader)
        else:
            header = mmap.mmap(
                hdrfile,
                0,
                access=mmap.ACCESS_READ,
                format=niftiheader[: (fileinfo.st_size < 352)],
            )
            nii["hdr"] = header[0]

    if nii["hdr"]["dim"][0] > 7:
        names = list(nii["hdr"].keys())
        for name in names:
            value = nii["hdr"][name]
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            # Swap bytes and change dtype
            nii["hdr"][name] = value.byteswap()

        if nii["hdr"]["sizeof_hdr"] > 540:
            nii["hdr"]["sizeof_hdr"] = nii["hdr"]["sizeof_hdr"].byteswap()

    type2byte = np.array(
        [
            [0, 0],  # unknown
            [1, 0],  # binary (1 bit/voxel)
            [2, 1],  # unsigned char (8 bits/voxel)
            [4, 2],  # signed short (16 bits/voxel)
            [8, 4],  # signed int (32 bits/voxel)
            [16, 4],  # float (32 bits/voxel)
            [32, 8],  # complex (64 bits/voxel)
            [64, 8],  # double (64 bits/voxel)
            [128, 3],  # RGB triple (24 bits/voxel)
            [255, 0],  # not very useful (?)
            [256, 1],  # signed char (8 bits)
            [512, 2],  # unsigned short (16 bits)
            [768, 4],  # unsigned int (32 bits)
            [1024, 8],  # long long (64 bits)
            [1280, 8],  # unsigned long long (64 bits)
            [1536, 16],  # long double (128 bits)
            [1792, 16],  # double pair (128 bits)
            [2048, 32],  # long double pair (256 bits)
            [2304, 4],  # 4 byte RGBA (32 bits/voxel)
        ],
        dtype=np.int32,
    )

    type2str = [
        ["uint8", 0],  # unknown
        ["uint8", 0],  # binary (1 bit/voxel)
        ["uint8", 1],  # unsigned char (8 bits/voxel)
        ["uint16", 1],  # signed short (16 bits/voxel)
        ["int32", 1],  # signed int (32 bits/voxel)
        ["float32", 1],  # float (32 bits/voxel)
        ["complex64", 2],  # complex (64 bits/voxel)
        ["float64", 1],  # double (64 bits/voxel)
        ["uint8", 3],  # RGB triple (24 bits/voxel)
        ["uint8", 0],  # not very useful (?)
        ["int8", 1],  # signed char (8 bits)
        ["uint16", 1],  # unsigned short (16 bits)
        ["uint32", 1],  # unsigned int (32 bits)
        ["int64", 1],  # long long (64 bits)
        ["uint64", 1],  # unsigned long long (64 bits)
        ["uint8", 16],  # long double (128 bits)
        ["uint8", 16],  # double pair (128 bits)
        ["uint8", 32],  # long double pair (256 bits)
        ["uint8", 4],  # 4 byte RGBA (32 bits/voxel)
    ]

    typeidx = np.where(type2byte[:, 0] == nii["hdr"]["datatype"])[0][0]

    nii["datatype"] = type2str[typeidx][0]
    nii["datalen"] = type2str[typeidx][1]
    nii["voxelbyte"] = type2byte[typeidx, 1]
    nii["endian"] = "little" if dataendian == "L" else "big"

    if type2byte[typeidx, 1] == 0:
        nii["img"] = []
        return nii

    if type2str[typeidx][1] > 1:
        nii["hdr"]["dim"] = np.hstack(
            (
                nii["hdr"]["dim"][0] + 1,
                np.uint16(nii["datalen"]),
                nii["hdr"]["dim"][1:-1],
            )
        )

    if len(varargin) > 0 and varargin[0] == "niiheader":
        return nii

    if re.search(r"\.[Hh][Dd][Rr](\.[Gg][Zz])*$", filename):
        filename = re.sub(r"\.[Hh][Dd][Rr](\.[Gg][Zz])*$", ".img\g<1>", filename)

    imgbytenum = (
        np.prod(nii["hdr"]["dim"][1 : nii["hdr"]["dim"][0] + 1]) * nii["voxelbyte"]
    )

    if isnii == 0 and re.search(r"\.[Gg][Zz]$", filename):
        with open(filename, "rb") as finput:
            input = finput.read()
        gzdata = zlib.decompress(bytes(input), zlib.MAX_WBITS | 32)
        nii["img"] = np.frombuffer(gzdata[:imgbytenum], dtype=nii["datatype"])
    else:
        if "gzdata" not in locals():
            with open(filename, "rb") as fid:
                if isnii:
                    fid.seek(int(nii["hdr"]["vox_offset"]))
                nii["img"] = np.frombuffer(fid.read(imgbytenum), dtype=nii["datatype"])
        else:
            nii["img"] = np.frombuffer(
                gzdata[
                    int(nii["hdr"]["vox_offset"][0]) : int(
                        nii["hdr"]["vox_offset"][0] + imgbytenum + 1
                    )
                ],
                dtype=nii["datatype"],
            )

    nii["img"] = nii["img"].reshape(nii["hdr"]["dim"][1 : nii["hdr"]["dim"][0] + 1])

    if len(varargin) > 0 and varargin[0] == "nii":
        return nii

    nii0 = nii.copy()

    nii = niiheader2jnii(nii0)

    nii["NIFTIData"] = nii0["img"]

    if "extension" in nii0["hdr"] and nii0["hdr"]["extension"][0] > 0:
        if "gzdata" in locals():
            nii["NIFTIExtension"] = []
            count = 0
            bufpos = nii0["hdr"]["sizeof_hdr"] + 4
            while bufpos < nii0["hdr"]["vox_offset"]:
                size = (
                    struct.unpack(dataendian + "I", gzdata[bufpos : bufpos + 4])[0] - 8
                )
                type = struct.unpack(dataendian + "I", gzdata[bufpos + 4 : bufpos + 8])[
                    0
                ]
                bufpos += 8
                if bufpos + size <= nii0["hdr"]["vox_offset"]:
                    nii["NIFTIExtension"].append(
                        {
                            "Size": size,
                            "Type": type,
                            "x0x5F_ByteStream_": gzdata[bufpos : bufpos + size],
                        }
                    )
                bufpos += size
                count += 1
        else:
            with open(filename, "rb") as fid:
                fid.seek(nii0["hdr"]["sizeof_hdr"] + 4)
                nii["NIFTIExtension"] = []
                count = 0
                while fid.tell() < nii0["hdr"]["vox_offset"]:
                    size = struct.unpack(dataendian + "I", fid.read(4))[0] - 8
                    type = struct.unpack(dataendian + "I", fid.read(4))[0]
                    if fid.tell() + size < nii0["hdr"]["vox_offset"]:
                        nii["NIFTIExtension"].append(
                            {
                                "Size": size,
                                "Type": type,
                                "x0x5F_ByteStream_": fid.read(size),
                            }
                        )
                    count += 1

    return nii


def niiformat(format):
    header = {
        "nifti1": [
            [
                "int32",
                [1],
                "sizeof_hdr",
            ],  # !< MUST be 348           %  % int sizeof_hdr;       %  ...
            [
                "int8",
                [10],
                "data_type",
            ],  # !< ++UNUSED++            %  % char data_type[10];   %  ...
            [
                "int8",
                [18],
                "db_name",
            ],  # !< ++UNUSED++            %  % char db_name[18];     %  ...
            [
                "int32",
                [1],
                "extents",
            ],  # !< ++UNUSED++            %  % int extents;          %  ...
            [
                "int16",
                [1],
                "session_error",
            ],  # !< ++UNUSED++            %  % short session_error;  %  ...
            [
                "int8",
                [1],
                "regular",
            ],  # !< ++UNUSED++            %  % char regular;         %  ...
            [
                "int8",
                [1],
                "dim_info",
            ],  # !< MRI slice ordering.   %  % char hkey_un0;        %  ...
            [
                "uint16",
                [8],
                "dim",
            ],  # !< Data array dimensions.%  % short dim[8];         %  ...
            [
                "single",
                [1],
                "intent_p1",
            ],  # !< 1st intent parameter. %  % short unused8/9;      %  ...
            [
                "single",
                [1],
                "intent_p2",
            ],  # !< 2nd intent parameter. %  % short unused10/11;    %  ...
            [
                "single",
                [1],
                "intent_p3",
            ],  # !< 3rd intent parameter. %  % short unused12/13;    %  ...
            [
                "int16",
                [1],
                "intent_code",
            ],  # !< NIFTI_INTENT_* code.  %  % short unused14;       %  ...
            [
                "int16",
                [1],
                "datatype",
            ],  # !< Defines data type!    %  % short datatype;       %  ...
            [
                "int16",
                [1],
                "bitpix",
            ],  # !< Number bits/voxel.    %  % short bitpix;         %  ...
            [
                "int16",
                [1],
                "slice_start",
            ],  # !< First slice index.    %  % short dim_un0;        %  ...
            [
                "single",
                [8],
                "pixdim",
            ],  # !< Grid spacings.        %  % float pixdim[8];      %  ...
            [
                "single",
                [1],
                "vox_offset",
            ],  # !< Offset into .nii file %  % float vox_offset;     %  ...
            [
                "single",
                [1],
                "scl_slope",
            ],  # !< Data scaling: slope.  %  % float funused1;       %  ...
            [
                "single",
                [1],
                "scl_inter",
            ],  # !< Data scaling: offset. %  % float funused2;       %  ...
            [
                "int16",
                [1],
                "slice_end",
            ],  # !< Last slice index.     %  % float funused3;       %  ...
            [
                "int8",
                [1],
                "slice_code",
            ],  # !< Slice timing order.   %                             ...
            [
                "int8",
                [1],
                "xyzt_units",
            ],  # !< Units of pixdim[1..4] %                             ...
            [
                "single",
                [1],
                "cal_max",
            ],  # !< Max display intensity %  % float cal_max;        %  ...
            [
                "single",
                [1],
                "cal_min",
            ],  # !< Min display intensity %  % float cal_min;        %  ...
            [
                "single",
                [1],
                "slice_duration",
            ],  # !< Time for 1 slice.     %  % float compressed;     %  ...
            [
                "single",
                [1],
                "toffset",
            ],  # !< Time axis shift.      %  % float verified;       %  ...
            [
                "int32",
                [1],
                "glmax",
            ],  # !< ++UNUSED++            %  % int glmax;            %  ...
            [
                "int32",
                [1],
                "glmin",
            ],  # !< ++UNUSED++            %  % int glmin;            %  ...
            [
                "int8",
                [80],
                "descrip",
            ],  # !< any text you like.    %  % char descrip[80];     %  ...
            [
                "int8",
                [24],
                "aux_file",
            ],  # !< auxiliary filename.   %  % char aux_file[24];    %  ...
            [
                "int16",
                [1],
                "qform_code",
            ],  # !< NIFTI_XFORM_* code.   %  %-- all ANALYZE 7.5 --- %  ...
            [
                "int16",
                [1],
                "sform_code",
            ],  # !< NIFTI_XFORM_* code.   %  %below here are replaced%  ...
            [
                "single",
                [1],
                "quatern_b",
            ],  # !< Quaternion b param.   %             ...
            [
                "single",
                [1],
                "quatern_c",
            ],  # !< Quaternion c param.   %             ...
            [
                "single",
                [1],
                "quatern_d",
            ],  # !< Quaternion d param.   %             ...
            [
                "single",
                [1],
                "qoffset_x",
            ],  # !< Quaternion x shift.   %             ...
            [
                "single",
                [1],
                "qoffset_y",
            ],  # !< Quaternion y shift.   %             ...
            [
                "single",
                [1],
                "qoffset_z",
            ],  # !< Quaternion z shift.   %             ...
            [
                "single",
                [4],
                "srow_x",
            ],  # !< 1st row affine transform.   %           ...
            [
                "single",
                [4],
                "srow_y",
            ],  # !< 2nd row affine transform.   %           ...
            [
                "single",
                [4],
                "srow_z",
            ],  # !< 3rd row affine transform.   %           ...
            [
                "int8",
                [16],
                "intent_name",
            ],  # !< 'name' or meaning of data.  %           ...
            ["int8", [4], "magic"],  # !< MUST be "ni1\0" or "n+1\0". %           ...
            ["int8", [4], "extension"],  # !< header extension      %             ...
        ],
        "nifti2": [
            [
                "int32",
                [1],
                "sizeof_hdr",
            ],  # !< MUST be 540           %  % int sizeof_hdr;       %  ...
            ["int8", [8], "magic"],  # !< MUST be "ni2\0" or "n+2\0". %           ...
            [
                "int16",
                [1],
                "datatype",
            ],  # !< Defines data type!    %  % short datatype;       %  ...
            [
                "int16",
                [1],
                "bitpix",
            ],  # !< Number bits/voxel.    %  % short bitpix;         %  ...
            [
                "int64",
                [8],
                "dim",
            ],  # !< Data array dimensions.%  % short dim[8];         %  ...
            [
                "double",
                [1],
                "intent_p1",
            ],  # !< 1st intent parameter. %  % short unused8/9;      %  ...
            [
                "double",
                [1],
                "intent_p2",
            ],  # !< 2nd intent parameter. %  % short unused10/11;    %  ...
            [
                "double",
                [1],
                "intent_p3",
            ],  # !< 3rd intent parameter. %  % short unused12/13;    %  ...
            [
                "double",
                [8],
                "pixdim",
            ],  # !< Grid spacings.        %  % float pixdim[8];      %  ...
            [
                "int64",
                [1],
                "vox_offset",
            ],  # !< Offset into .nii file %  % float vox_offset;     %  ...
            [
                "double",
                [1],
                "scl_slope",
            ],  # !< Data scaling: slope.  %  % float funused1;       %  ...
            [
                "double",
                [1],
                "scl_inter",
            ],  # !< Data scaling: offset. %  % float funused2;       %  ...
            [
                "double",
                [1],
                "cal_max",
            ],  # !< Max display intensity %  % float cal_max;        %  ...
            [
                "double",
                [1],
                "cal_min",
            ],  # !< Min display intensity %  % float cal_min;        %  ...
            [
                "double",
                [1],
                "slice_duration",
            ],  # !< Time for 1 slice.     %  % float compressed;     %  ...
            [
                "double",
                [1],
                "toffset",
            ],  # !< Time axis shift.      %  % float verified;       %  ...
            [
                "int64",
                [1],
                "slice_start",
            ],  # !< First slice index.    %  % short dim_un0;        %  ...
            [
                "int64",
                [1],
                "slice_end",
            ],  # !< Last slice index.     %  % float funused3;       %  ...
            [
                "int8",
                [80],
                "descrip",
            ],  # !< any text you like.    %  % char descrip[80];     %  ...
            [
                "int8",
                [24],
                "aux_file",
            ],  # !< auxiliary filename.   %  % char aux_file[24];    %  ...
            [
                "int32",
                [1],
                "qform_code",
            ],  # !< NIFTI_XFORM_* code.   %  %-- all ANALYZE 7.5 --- %  ...
            [
                "int32",
                [1],
                "sform_code",
            ],  # !< NIFTI_XFORM_* code.   %  %below here are replaced%  ...
            [
                "double",
                [1],
                "quatern_b",
            ],  # !< Quaternion b param.   %             ...
            [
                "double",
                [1],
                "quatern_c",
            ],  # !< Quaternion c param.   %             ...
            [
                "double",
                [1],
                "quatern_d",
            ],  # !< Quaternion d param.   %             ...
            [
                "double",
                [1],
                "qoffset_x",
            ],  # !< Quaternion x shift.   %             ...
            [
                "double",
                [1],
                "qoffset_y",
            ],  # !< Quaternion y shift.   %             ...
            [
                "double",
                [1],
                "qoffset_z",
            ],  # !< Quaternion z shift.   %             ...
            [
                "double",
                [4],
                "srow_x",
            ],  # !< 1st row affine transform.   %           ...
            [
                "double",
                [4],
                "srow_y",
            ],  # !< 2nd row affine transform.   %           ...
            [
                "double",
                [4],
                "srow_z",
            ],  # !< 3rd row affine transform.   %           ...
            [
                "int32",
                [1],
                "slice_code",
            ],  # !< Slice timing order.   %                             ...
            [
                "int32",
                [1],
                "xyzt_units",
            ],  # !< Units of pixdim[1..4] %                             ...
            [
                "int32",
                [1],
                "intent_code",
            ],  # !< NIFTI_INTENT_* code.  %  % short unused14;       %  ...
            [
                "int8",
                [16],
                "intent_name",
            ],  # !< 'name' or meaning of data.  %           ...
            [
                "int8",
                [1],
                "dim_info",
            ],  # !< MRI slice ordering.   %  % char hkey_un0;        %  ...
            ["int8", [15], "reserved"],  # !< unused buffer     %             ...
            ["int8", [4], "extension"],  # !< header extension      %             ...
        ],
    }

    if format == "":
        format = "nifti1"

    format = format.lower()

    if format in header:
        niiheader = header[format]
    else:
        raise ValueError("format must be either nifti1 or nifti2")

    return niiheader


def niicodemap(name, value):
    """
    Convert between NIFTI numeric codes and human-readable string header values.

    Parameters
    ----------
    name : str
        The NIFTI field name. Supports:
        'intent_code', 'slice_code', 'datatype', 'qform_code',
        'sform_code', 'xyzt_units', 'unit', 'Intent', 'SliceType',
        'DataType', 'QForm', 'SForm'
    value : str or int
        A string name or numeric code to convert.

    Returns
    -------
    newval : int or str
        Mapped value in the opposite domain of the input.
    """

    # Lookup table: code -> name
    lut = defaultdict(dict)
    lut["intent_code"] = {
        0: "",
        2: "corr",
        3: "ttest",
        4: "ftest",
        5: "zscore",
        6: "chi2",
        7: "beta",
        8: "binomial",
        9: "gamma",
        10: "poisson",
        11: "normal",
        12: "ncftest",
        13: "ncchi2",
        14: "logistic",
        15: "laplace",
        16: "uniform",
        17: "ncttest",
        18: "weibull",
        19: "chi",
        20: "invgauss",
        21: "extval",
        22: "pvalue",
        23: "logpvalue",
        24: "log10pvalue",
        1001: "estimate",
        1002: "label",
        1003: "neuronames",
        1004: "matrix",
        1005: "symmatrix",
        1006: "dispvec",
        1007: "vector",
        1008: "point",
        1009: "triangle",
        1010: "quaternion",
        1011: "unitless",
        2001: "tseries",
        2002: "elem",
        2003: "rgb",
        2004: "rgba",
        2005: "shape",
        2006: "fsl_fnirt_displacement_field",
        2007: "fsl_cubic_spline_coefficients",
        2008: "fsl_dct_coefficients",
        2009: "fsl_quadratic_spline_coefficients",
        2016: "fsl_topup_cubic_spline_coefficients",
        2017: "fsl_topup_quadratic_spline_coefficients",
        2018: "fsl_topup_field",
    }

    lut["slice_code"] = {
        0: "",
        1: "seq+",
        2: "seq-",
        3: "alt+",
        4: "alt-",
        5: "alt2+",
        6: "alt-",
    }

    lut["datatype"] = {
        0: "",
        2: "uint8",
        4: "int16",
        8: "int32",
        16: "single",
        32: "complex64",
        64: "double",
        128: "rgb24",
        256: "int8",
        512: "uint16",
        768: "uint32",
        1024: "int64",
        1280: "uint64",
        1536: "double128",
        1792: "complex128",
        2048: "complex256",
        2304: "rgba32",
    }

    lut["xyzt_units"] = {
        0: "",
        1: "m",
        2: "mm",
        3: "um",
        8: "s",
        16: "ms",
        24: "us",
        32: "hz",
        40: "ppm",
        48: "rad",
    }

    lut["qform_code"] = {
        0: "",
        1: "scanner_anat",
        2: "aligned_anat",
        3: "talairach",
        4: "mni_152",
        5: "template_other",
    }

    # Aliases for consistency
    lut["sform_code"] = lut["qform_code"]
    lut["unit"] = lut["xyzt_units"]
    lut["slicetype"] = lut["slice_code"]
    lut["intent"] = lut["intent_code"]
    lut["Intent"] = lut["intent_code"]
    lut["SliceType"] = lut["slice_code"]
    lut["DataType"] = lut["datatype"]
    lut["QForm"] = lut["qform_code"]
    lut["SForm"] = lut["sform_code"]
    lut["Unit"] = lut["xyzt_units"]

    name = name.lower()

    if name not in lut:
        raise ValueError(f"Unsupported field name: {name}")

    if isinstance(value, (np.ndarray, np.generic)) and value.size == 1:
        if value.ndim == 0:
            value = int(value)
        else:
            value = int(value[0])

    if isinstance(value, (int, float)):
        value = int(value)
        if value not in lut[name]:
            raise ValueError(f"Code {value} not found in {name}")
        return lut[name][value]

    # Reverse LUT: name -> code
    rev_lut = {v: k for k, v in lut[name].items()}

    if value not in rev_lut:
        raise ValueError(f"String value '{value}' not found in {name}")
    return rev_lut[value]


def niiheader2jnii(nii0):
    nii = defaultdict()
    nii["NIFTIHeader"] = defaultdict()
    nii["NIFTIHeader"]["NIIHeaderSize"] = nii0["hdr"]["sizeof_hdr"]
    if "data_type" in nii0["hdr"]:
        nii["NIFTIHeader"]["A75DataTypeName"] = "".join(
            map(chr, nii0["hdr"]["data_type"])
        ).rstrip("\x00")
        nii["NIFTIHeader"]["A75DBName"] = "".join(
            map(chr, nii0["hdr"]["db_name"])
        ).rstrip("\x00")
        nii["NIFTIHeader"]["A75Extends"] = nii0["hdr"]["extents"]
        nii["NIFTIHeader"]["A75SessionError"] = nii0["hdr"]["session_error"]
        nii["NIFTIHeader"]["A75Regular"] = nii0["hdr"]["regular"]
    nii["NIFTIHeader"]["DimInfo"] = defaultdict()
    nii["NIFTIHeader"]["DimInfo"]["Freq"] = nii0["hdr"]["dim_info"] & 7
    nii["NIFTIHeader"]["DimInfo"]["Phase"] = (nii0["hdr"]["dim_info"] >> 3) & 7
    nii["NIFTIHeader"]["DimInfo"]["Slice"] = (nii0["hdr"]["dim_info"] >> 6) & 7
    nii["NIFTIHeader"]["Dim"] = nii0["hdr"]["dim"][1 : 1 + nii0["hdr"]["dim"][0]]
    nii["NIFTIHeader"]["Param1"] = nii0["hdr"]["intent_p1"]
    nii["NIFTIHeader"]["Param2"] = nii0["hdr"]["intent_p2"]
    nii["NIFTIHeader"]["Param3"] = nii0["hdr"]["intent_p3"]
    nii["NIFTIHeader"]["Intent"] = niicodemap("intent", nii0["hdr"]["intent_code"])
    nii["NIFTIHeader"]["DataType"] = niicodemap("datatype", nii0["hdr"]["datatype"])
    nii["NIFTIHeader"]["BitDepth"] = nii0["hdr"]["bitpix"]
    nii["NIFTIHeader"]["FirstSliceID"] = nii0["hdr"]["slice_start"]
    nii["NIFTIHeader"]["VoxelSize"] = nii0["hdr"]["pixdim"][
        1 : 1 + nii0["hdr"]["dim"][0]
    ]
    nii["NIFTIHeader"]["Orientation"] = {"x": "r", "y": "a", "z": "s"}
    if nii0["hdr"]["pixdim"][0] < 0:
        nii["NIFTIHeader"]["Orientation"] = {"x": "l", "y": "a", "z": "s"}
    nii["NIFTIHeader"]["NIIByteOffset"] = nii0["hdr"]["vox_offset"]
    nii["NIFTIHeader"]["ScaleSlope"] = nii0["hdr"]["scl_slope"]
    nii["NIFTIHeader"]["ScaleOffset"] = nii0["hdr"]["scl_inter"]
    nii["NIFTIHeader"]["LastSliceID"] = nii0["hdr"]["slice_end"]
    nii["NIFTIHeader"]["SliceType"] = niicodemap("slicetype", nii0["hdr"]["slice_code"])
    nii["NIFTIHeader"]["Unit"] = defaultdict()
    nii["NIFTIHeader"]["Unit"]["L"] = niicodemap("unit", nii0["hdr"]["xyzt_units"] & 7)
    nii["NIFTIHeader"]["Unit"]["T"] = niicodemap(
        "unit", (nii0["hdr"]["xyzt_units"] >> 3) & 7
    )
    nii["NIFTIHeader"]["MaxIntensity"] = nii0["hdr"]["cal_max"]
    nii["NIFTIHeader"]["MinIntensity"] = nii0["hdr"]["cal_min"]
    nii["NIFTIHeader"]["SliceTime"] = nii0["hdr"]["slice_duration"]
    nii["NIFTIHeader"]["TimeOffset"] = nii0["hdr"]["toffset"]
    if "glmax" in nii0["hdr"]:
        nii["NIFTIHeader"]["A75GlobalMax"] = nii0["hdr"]["glmax"]
        nii["NIFTIHeader"]["A75GlobalMin"] = nii0["hdr"]["glmin"]
    nii["NIFTIHeader"]["Description"] = "".join(
        map(chr, nii0["hdr"]["descrip"])
    ).rstrip("\x00")
    nii["NIFTIHeader"]["AuxFile"] = "".join(map(chr, nii0["hdr"]["aux_file"])).rstrip(
        "\x00"
    )
    nii["NIFTIHeader"]["QForm"] = nii0["hdr"]["qform_code"]
    nii["NIFTIHeader"]["SForm"] = nii0["hdr"]["sform_code"]
    nii["NIFTIHeader"]["Quatern"] = defaultdict()
    nii["NIFTIHeader"]["Quatern"]["b"] = nii0["hdr"]["quatern_b"]
    nii["NIFTIHeader"]["Quatern"]["c"] = nii0["hdr"]["quatern_c"]
    nii["NIFTIHeader"]["Quatern"]["d"] = nii0["hdr"]["quatern_d"]
    nii["NIFTIHeader"]["QuaternOffset"] = defaultdict()
    nii["NIFTIHeader"]["QuaternOffset"]["x"] = nii0["hdr"]["qoffset_x"]
    nii["NIFTIHeader"]["QuaternOffset"]["y"] = nii0["hdr"]["qoffset_y"]
    nii["NIFTIHeader"]["QuaternOffset"]["z"] = nii0["hdr"]["qoffset_z"]
    nii["NIFTIHeader"]["Affine"] = np.vstack(
        [
            nii0["hdr"]["srow_x"],
            nii0["hdr"]["srow_y"],
            nii0["hdr"]["srow_z"],
        ]
    )
    nii["NIFTIHeader"]["Name"] = "".join(map(chr, nii0["hdr"]["intent_name"])).rstrip(
        "\x00"
    )
    nii["NIFTIHeader"]["NIIFormat"] = "".join(map(chr, nii0["hdr"]["magic"])).rstrip(
        "\x00"
    )
    if "extension" in nii0["hdr"]:
        nii["NIFTIHeader"]["NIIExtender"] = nii0["hdr"]["extension"]
    nii["NIFTIHeader"]["NIIQfac_"] = nii0["hdr"]["pixdim"][0]
    nii["NIFTIHeader"]["NIIEndian_"] = "little"
    if "endian" in nii0:
        nii["NIFTIHeader"]["NIIEndian_"] = nii0["endian"]
    if "reserved" in nii0["hdr"]:
        nii["NIFTIHeader"]["NIIUnused_"] = nii0["hdr"]["reserved"]

    return nii


def jnii2nii(jnii, niifile=None):
    """
    Covert a JNIfTI file or data structure to a NIfTI-1/2 structure or file

    Args:
        jnii: a JNIfTI data structure (a dict with 'NIFTIHeader' and 'NIFTIData' keys);
              if jnii is a string, it represents a JNIfTI file (.jnii/.bnii)
        niifile: if the 2nd parameter is given as a file name, the converted nifti data
              will be save as a nii file with filename specified by niifile.
              if the filename in niifile contains .gz, the file will be compressed using
              the zmat toolbox.

    Returns:
        nii: is the converted nifti-1/2 data structure, it contains the below subfields
          nii['img']: the data volume read from the nii file
          nii['hdr']: extended raw file header, a dict that is byte-wise compatible with a
                   nifti-1 - in this case, typecast(nii['hdr'],'uint8') must be 348+4=352 bytes,
                       including the raw nifti-1 hdr header (348 bytes) plus the 4-byte
                       extension flags), or
                   nifti-2 - in this case, typecast(nii['hdr'],'uint8') must be 540+4=544 bytes,
                       including the raw nifti-2 hdr header (540 bytes) plus the 4-byte
                       extension flags)
              if one run nii['hdr']['extension']=[]; the resulting dict is 348/540-byte in length
              nii['hdr'] key subfields include

              sizeof_hdr: must be 348 (for NIFTI-1) or 540 (for NIFTI-2)
              dim: list, dim[1] defines the array size
              datatype: the type of data stored in each voxel
              bitpix: total bits per voxel
              magic: must be 'ni1\0' or 'n+1\0' for NIFTI-1 data, and 'ni2\0' or 'n+2\0' for NIFTI-2 data

              For the detailed nii header, please see
              https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    """

    if jnii is None:
        raise ValueError("jnii input must be provided")

    if isinstance(jnii, str):
        jnii = loadjnifti(jnii)

    hdr = jnii["NIFTIHeader"].copy()

    if not isinstance(jnii, dict):
        raise ValueError(
            "input must be a valid JNIfTI structure (needs both NIFTIHeader and NIFTIData subfields)"
        )

    niiformat = "nifti1"

    if (
        "NIIFormat" in hdr
        and hdr["NIIFormat"][:3] in ["ni2", "n+2"]
        or np.max(hdr["Dim"]) >= 2**32
    ):
        niiformat = "nifti2"

    nii = nifticreate(jnii["NIFTIData"], niiformat)

    if "NIIHeaderSize" in hdr:
        nii["hdr"]["sizeof_hdr"] = bytematch(
            hdr, "NIIHeaderSize", nii["hdr"]["sizeof_hdr"]
        )
    if "data_type" in nii["hdr"]:
        nii["hdr"]["data_type"] = bytematch(
            hdr, "A75DataTypeName", nii["hdr"]["data_type"]
        )
        nii["hdr"]["db_name"] = bytematch(hdr, "A75DBName", nii["hdr"]["db_name"])
        nii["hdr"]["extents"] = bytematch(hdr, "A75Extends", nii["hdr"]["extents"])
        nii["hdr"]["session_error"] = bytematch(
            hdr, "A75SessionError", nii["hdr"]["session_error"]
        )
        nii["hdr"]["regular"] = bytematch(hdr, "A75Regular", nii["hdr"]["regular"])

    dim_info = np.bitwise_or(
        np.uint8(hdr["DimInfo"]["Freq"]),
        np.bitwise_or(
            np.left_shift(np.uint8(hdr["DimInfo"]["Phase"]), 3),
            np.left_shift(np.uint8(hdr["DimInfo"]["Slice"]), 6),
        ),
    )
    nii["hdr"]["dim_info"] = dim_info.astype(nii["hdr"]["dim_info"].dtype)

    nii["hdr"]["dim"][0] = len(hdr["Dim"])
    nii["hdr"]["dim"][1 : 1 + len(hdr["Dim"])] = bytematch(
        hdr,
        "Dim",
        nii["hdr"]["dim"][1 : 1 + len(hdr["Dim"])],
    )
    nii["hdr"]["intent_p1"] = bytematch(hdr, "Param1", nii["hdr"]["intent_p1"])
    nii["hdr"]["intent_p2"] = bytematch(hdr, "Param2", nii["hdr"]["intent_p2"])
    nii["hdr"]["intent_p3"] = bytematch(hdr, "Param3", nii["hdr"]["intent_p3"])

    if isinstance(hdr["Intent"], str):
        hdr["Intent"] = niicodemap("intent", hdr["Intent"])
    nii["hdr"]["intent_code"] = bytematch(hdr, "Intent", nii["hdr"]["intent_code"])
    if isinstance(hdr["DataType"], str):
        hdr["DataType"] = niicodemap("datatype", hdr["DataType"])

    nii["hdr"]["datatype"] = bytematch(hdr, "DataType", nii["hdr"]["datatype"])
    nii["hdr"]["bitpix"] = bytematch(hdr, "BitDepth", nii["hdr"]["bitpix"])
    nii["hdr"]["slice_start"] = bytematch(
        hdr, "FirstSliceID", nii["hdr"]["slice_start"]
    )
    nii["hdr"]["pixdim"][0] = len(hdr["VoxelSize"])
    nii["hdr"]["pixdim"][1 : 1 + nii["hdr"]["dim"][0] - 1] = bytematch(
        hdr,
        "VoxelSize",
        nii["hdr"]["pixdim"][1 : 1 + nii["hdr"]["dim"][0] - 1],
    )
    nii["hdr"]["vox_offset"] = bytematch(hdr, "NIIByteOffset", nii["hdr"]["vox_offset"])
    nii["hdr"]["scl_slope"] = bytematch(hdr, "ScaleSlope", nii["hdr"]["scl_slope"])
    nii["hdr"]["scl_inter"] = bytematch(hdr, "ScaleOffset", nii["hdr"]["scl_inter"])
    nii["hdr"]["slice_end"] = bytematch(hdr, "LastSliceID", nii["hdr"]["slice_end"])

    if isinstance(hdr["SliceType"], str):
        hdr["SliceType"] = niicodemap("slicetype", hdr["SliceType"])
    nii["hdr"]["slice_code"] = bytematch(hdr, "SliceType", nii["hdr"]["slice_code"])
    if isinstance(hdr["Unit"]["L"], str):
        hdr["Unit"]["L"] = niicodemap("unit", hdr["Unit"]["L"])
    if isinstance(hdr["Unit"]["T"], str):
        hdr["Unit"]["T"] = niicodemap("unit", hdr["Unit"]["T"])

    xyzt_units = np.bitwise_or(
        np.uint8(hdr["Unit"]["L"]),
        np.uint8(hdr["Unit"]["T"]),
    )
    nii["hdr"]["xyzt_units"] = xyzt_units.astype(nii["hdr"]["xyzt_units"].dtype)

    nii["hdr"]["cal_max"] = bytematch(hdr, "MaxIntensity", nii["hdr"]["cal_max"])
    nii["hdr"]["cal_min"] = bytematch(hdr, "MinIntensity", nii["hdr"]["cal_min"])
    nii["hdr"]["slice_duration"] = bytematch(
        hdr, "SliceTime", nii["hdr"]["slice_duration"]
    )
    nii["hdr"]["toffset"] = bytematch(hdr, "TimeOffset", nii["hdr"]["toffset"])
    if "glmax" in nii["hdr"]:
        nii["hdr"]["glmax"] = bytematch(hdr, "A75GlobalMax", nii["hdr"]["glmax"])
        nii["hdr"]["glmin"] = bytematch(hdr, "A75GlobalMin", nii["hdr"]["glmin"])

    nii["hdr"]["descrip"] = bytematch(hdr, "Description", nii["hdr"]["descrip"])
    nii["hdr"]["aux_file"] = bytematch(hdr, "AuxFile", nii["hdr"]["aux_file"])

    nii["hdr"]["qform_code"] = bytematch(hdr, "QForm", nii["hdr"]["qform_code"])
    nii["hdr"]["sform_code"] = bytematch(hdr, "SForm", nii["hdr"]["sform_code"])
    nii["hdr"]["quatern_b"] = bytematch(hdr["Quatern"], "b", nii["hdr"]["quatern_b"])
    nii["hdr"]["quatern_c"] = bytematch(hdr["Quatern"], "c", nii["hdr"]["quatern_c"])
    nii["hdr"]["quatern_d"] = bytematch(hdr["Quatern"], "d", nii["hdr"]["quatern_d"])
    nii["hdr"]["qoffset_x"] = bytematch(
        hdr["QuaternOffset"], "x", nii["hdr"]["qoffset_x"]
    )
    nii["hdr"]["qoffset_y"] = bytematch(
        hdr["QuaternOffset"], "y", nii["hdr"]["qoffset_y"]
    )
    nii["hdr"]["qoffset_z"] = bytematch(
        hdr["QuaternOffset"], "z", nii["hdr"]["qoffset_z"]
    )

    affine = np.array(hdr["Affine"])
    nii["hdr"]["srow_x"] = affine[0].astype(nii["hdr"]["srow_x"].dtype)
    nii["hdr"]["srow_y"] = affine[1].astype(nii["hdr"]["srow_y"].dtype)
    nii["hdr"]["srow_z"] = affine[2].astype(nii["hdr"]["srow_z"].dtype)

    nii["hdr"]["intent_name"] = bytematch(hdr, "Name", nii["hdr"]["intent_name"])
    nii["hdr"]["magic"] = bytematch(hdr, "NIIFormat", nii["hdr"]["magic"])

    if "NIIExtender" in hdr:
        nii["hdr"]["extension"] = bytematch(hdr, "NIIExtender", nii["hdr"]["extension"])
    if "NIIQfac_" in hdr:
        nii["hdr"]["pixdim"][0] = bytematch(hdr, "NIIQfac_", nii["hdr"]["pixdim"][0])
    if "NIIUnused_" in hdr:
        nii["hdr"]["reserved"] = bytematch(hdr, "NIIUnused_", nii["hdr"]["reserved"])

    if "NIFTIExtension" in jnii and isinstance(jnii["NIFTIExtension"], list):
        nii["extension"] = jnii["NIFTIExtension"]
        if nii["hdr"]["extension"][0] != len(jnii["NIFTIExtension"]):
            nii["hdr"]["extension"][0] = len(jnii["NIFTIExtension"])
            warnings.warn(
                "header extension count does not match the extension data, force update"
            )

    if niifile is not None:
        savenifti(nii["img"], niifile, nii["hdr"])

    return nii


def bytematch(jobj, key, orig):
    """
    Match the field `key` in dictionary `jobj` with the same data type and length as `orig`.

    Parameters
    ----------
    jobj : dict
        Input dictionary representing a JSON-like object.
    key : str
        Field name to look for in `jobj`.
    orig : np.ndarray
        Original numpy array whose type and shape should be preserved.

    Returns
    -------
    dat : np.ndarray
        Output array matching the type and shape of `orig`, using values from `jobj[key]` if available.
    """
    dtype = orig.dtype
    dat = orig.copy()

    if key in jobj:
        dat = (
            np.frombuffer(jobj[key].encode(), dtype=dtype)
            if isinstance(jobj[key], str)
            else np.array(jobj[key], dtype=dtype)
        )
    else:
        dat = np.array([0], dtype=dtype)

    # Pad or trim the result to match the original length
    if dat.size < orig.size:
        # Extend with zeros
        padded = np.zeros(orig.size, dtype=dtype)
        padded[: dat.size] = dat
        dat = padded
    elif dat.size > orig.size:
        dat = dat[: orig.size]

    return dat


def nifticreate(img, format="nifti1", niihdr0=None, **kwargs):
    """
    Create a default NIfTI header from an image array.

    Parameters
    ----------
    img : numpy.ndarray
        The image data array matching the header to create.
    format : str, optional
        Format specifier, only 'nifti1' supported (default).

    Returns
    -------
    header : dict
        Dictionary representing the NIfTI header.
    """

    # Map Python/numpy dtypes to NIfTI codes
    datatype_map = {
        "int8": 256,
        "int16": 4,
        "int32": 8,
        "int64": 1024,
        "uint8": 2,
        "uint16": 512,
        "uint32": 768,
        "uint64": 1280,
        "float32": 16,
        "float64": 64,
    }

    if format == "nifti1":
        headerlen = 348
    else:
        headerlen = 540

    # Create empty header
    rawbytes = np.zeros(headerlen + 4, dtype=np.uint8)
    header = memmapstream(rawbytes, niiformat(format))

    niihdr = None
    if niihdr0 is not None:
        niihdr = niihdr0

    if isinstance(img, dict) and "hdr" in img and "img" in img:
        if niihdr is None:
            niihdr = img["hdr"]
        img = img["img"]

    if niihdr is not None:
        for k in niihdr:
            if k in header:
                header[k] = bytematch(niihdr, k, header[k])

    # Set values
    header["sizeof_hdr"] = np.array(headerlen, dtype=np.int32)
    np_dtype_str = str(img.dtype)

    if np_dtype_str not in datatype_map:
        raise ValueError(f"Unsupported image data type: {np_dtype_str}")
    header["datatype"] = np.array(
        datatype_map[np_dtype_str], dtype=header["datatype"].dtype
    )

    # Fill dim and pixdim
    ndim = img.ndim
    shape = img.shape
    header["dim"] = np.ones(8, dtype=type(header["dim"][0]))
    header["dim"][0] = ndim
    header["dim"][1 : ndim + 1] = shape
    header["pixdim"] = np.ones(8, dtype=type(header["pixdim"][0]))

    header["vox_offset"] = np.array(headerlen + 4, dtype=header["vox_offset"].dtype)

    # Set magic
    if headerlen == 540:
        header["magic"] = np.frombuffer(
            b"ni2\x00\x00\x00\x00\x00", dtype=type(header["magic"][0])
        )
    else:
        header["magic"] = np.frombuffer(b"ni1\x00", dtype=type(header["magic"][0]))

    # Set affine transform identity matrix
    header["srow_x"] = np.array([1, 0, 0, 0], dtype=type(header["srow_x"][0]))
    header["srow_y"] = np.array([0, 1, 0, 0], dtype=type(header["srow_y"][0]))
    header["srow_z"] = np.array([0, 0, 1, 0], dtype=type(header["srow_z"][0]))
    header["sform_code"] = np.array(1, dtype=header["sform_code"].dtype)

    if kwargs.get("headeronly", False):
        return header

    nii = defaultdict()
    nii["hdr"] = header
    nii["img"] = img
    return nii


def jnifticreate(*args, **kwargs):
    """
    Create a default JNIfTI structure with default header and image volume

    Args:
        *args: set the jnii.NIFTIData section
        **kwargs: the header subfield name defined in the JNIfTI
                  specification, see https://github.com/NeuroJSON/jnifti

    Returns:
        jnii: without any input, jnii gives the default jnii header
              if img is given, jnii also includes the NIFTIData field
    """
    jnii = {
        "_DataInfo_": {
            "JNIFTIVersion": "0.5",
            "Comment": "Created by JNIFTY Toolbox (https://github.com/NeuroJSON/jnifty)",
            "AnnotationFormat": "https://github.com/NeuroJSON/jnifti/blob/master/JNIfTI_specification.md",
            "SerialFormat": "https://json.org",
            "Parser": {
                "Python": [],
                "MATLAB": [],
                "JavaScript": "https://github.com/NeuroJSON/jsdata",
                "CPP": "https://github.com/NeuroJSON/json",
                "C": "https://github.com/NeuroJSON/ubj",
            },
        }
    }
    jnii["_DataInfo_"]["Parser"]["Python"] = [
        "https://pypi.org/project/jdata",
        "https://pypi.org/project/bjdata",
    ]
    jnii["_DataInfo_"]["Parser"]["MATLAB"] = [
        "https://github.com/NeuroJSON/jnifty",
        "https://github.com/NeuroJSON/jsonlab",
    ]

    if len(args) == 0:
        return jnii

    img = None
    pid = 0
    if not isinstance(args[0], str):
        img = args[0]
        pid = 1

    if len(args) > pid:
        for i in range(pid, len(args), 2):
            jnii["NIFTIHeader"][args[i]] = args[i + 1]

    if img is not None:
        if not isinstance(img, (np.ndarray, np.generic)):
            raise ValueError("img input must be a numerical or logical array")
        jnii["NIFTIHeader"]["Dim"] = img.shape
        jnii["NIFTIHeader"]["DataType"] = img.dtype.name
        info = (
            np.iinfo(img.dtype)
            if issubclass(img.dtype.type, np.integer)
            else np.finfo(img.dtype)
        )
        jnii["NIFTIHeader"]["BitDepth"] = info.bits
        jnii["NIFTIHeader"]["MinIntensity"] = np.min(img)
        jnii["NIFTIHeader"]["MaxIntensity"] = np.max(img)
        jnii["NIFTIData"] = img

    return jnii


def loadjnifti(filename, *args, **kwargs):
    """
    Load a standard NIFTI-1/2 file or text or binary JNIfTI file with
    format defined in JNIfTI specification: https://github.com/NeuroJSON/jnifti

    Parameters:
    filename (str): The input file name to the JNIfTI or NIFTI-1/2 file
                    *.bnii for binary JNIfTI file
                    *.jnii for text JNIfTI file
                    *.nii  for NIFTI-1/2 files
    *args: Optional arguments. If loading from a .bnii file, please see the options for
           loadbj (part of JSONLab); if loading from a .jnii, please see the
           supported options for loadjson (part of JSONLab).

    Returns:
    jnii (dict or list): A structure (array) or cell (array). The data structure can
        be completely generic or auxilary data without any JNIfTI
        constructs. However, if a JNIfTI object is included, it shall
        contain the below subfields (can appear within any depth of the
        structure)
            jnii['NIFTIHeader'] -  a dict containing the 1-to-1 mapped NIFTI-1/2 header
            jnii['NIFTIData'] - the main image data array
            jnii['NIFTIExtension'] - a list containing the extension data buffers
    """
    if not filename.endswith((".nii", ".jnii", ".bnii")):
        raise ValueError(
            "File suffix must be .jnii for text JNIfTI, .bnii for binary JNIfTI or .nii for NIFTI-1/2 files"
        )

    if filename.endswith(".nii"):
        nii = loadnifti(filename, **kwargs)
    elif filename.endswith(".jnii"):
        # Assuming loadjson is available from JSONLab
        jnii = jd.load(filename, *args, **kwargs)
    elif filename.endswith(".bnii"):
        # Assuming loadbj is available from JSONLab
        jnii = jd.load(filename, *args, **kwargs)

    return jnii


def loadnifti(*args, **kwargs):
    """
    Alias for nii2jnii
    """
    return nii2jnii(*args, **kwargs)


def savenifti(img, filename, *args, **kwargs):
    """
    Write an image to a NIfTI (*.nii) or compressed NIfTI file (.nii.gz)

    Parameters:
    img (np.ndarray): Numerical array to be stored in the NIfTI file
    filename (str): Output file name, can have a suffix of '.nii' or '.nii.gz'
    *args: Optional arguments.
        If the first argument is a dict, it is treated as a pre-created/loaded NIfTI header data structure.
        If the first argument is a string 'nifti1' or 'nifti2', this function calls
        nifticreate to create a default header.

    Returns:
    bytestream (bytes, optional): The output file byte stream if no filename is given.
    """
    if args:
        if isinstance(args[0], (defaultdict, dict)):
            header = args[0]
        elif args[0] in ("nifti1", "nifti2"):
            header = nifticreate(img, args[0], headeronly=True, **kwargs)
    else:
        header = nifticreate(img, headeronly=True, **kwargs)

    buf = b"".join(header[name].tobytes() for name in header)

    if len(buf) not in (352, 544):
        raise ValueError(f"Incorrect nifti-1/2 header {len(buf)}")

    buf += img.tobytes()

    if len(args) > 1 and not filename:
        return buf

    if filename.endswith(".gz"):
        gzipper = zlib.compressobj(wbits=(zlib.MAX_WBITS | 16))
        buf = gzipper.compress(buf)
        buf += gzipper.flush()

    with open(filename, "wb") as f:
        f.write(buf)


def savejnifti(jnii, filename, *args):
    """
    Save an in-memory JNIfTI structure into a JNIfTI file with format
    defined in JNIfTI specification: https://github.com/NeuroJSON/jnifti

    Parameters:
    jnii (dict or list): A structure (array) or cell (array). The data structure can
        be completely generic or auxilary data without any JNIfTI
        constructs. However, if a JNIfTI object is included, it shall
        contain the below subfields (can appear within any depth of the
        structure)
            jnii['NIFTIHeader'] -  a dict containing the 1-to-1 mapped NIFTI-1/2 header
            jnii['NIFTIData'] - the main image data array
            jnii['NIFTIExtension'] - a list containing the extension data buffers
    filename (str): The output file name to the JNIfTI file
                    *.bnii for binary JNIfTI file
                    *.jnii for text JNIfTI file
    *args: Optional arguments. If saving to a .bnii file, please see the options for
           savebj (part of JSONLab); if saving to .jnii, please see the
           supported options for savejson (part of JSONLab).
    """
    if not filename.endswith((".jnii", ".bnii")):
        raise ValueError(
            "File suffix must be .jnii for text JNIfTI or .bnii for binary JNIfTI"
        )

    if filename.endswith(".jnii"):
        # Assuming savejnii is available from JSONLab
        jd.save(jnii, filename, *args)
    elif filename.endswith(".bnii"):
        # Assuming savebnii is available from JSONLab
        jd.save(jnii, filename, *args)


def memmapstream(bytes_in: Union[bytes, bytearray, np.ndarray], format: list):
    """
    Map a byte stream into structured data using a format specification.

    Parameters
    ----------
    bytes_in : bytes, bytearray, or numpy.ndarray of dtype uint8/int8
        Input byte stream
    format : list of [dtype_str, shape_tuple, field_name]
        Specification of fields. Each item is:
            - dtype_str: a string like 'int8', 'float32', etc.
            - shape_tuple: tuple of shape dimensions (e.g. (1, 8))
            - field_name: the name of the output dictionary key

    Returns
    -------
    outstruct : dict
        Dictionary mapping field names to reshaped numpy arrays

    Example
    -------
    b = bytearray(b'Andy') + bytearray([5]) + bytearray(b'JT')
    fmt = [
        ['uint8', (4), 'name'],
        ['uint8', (1), 'age'],
        ['uint8', (2), 'school']
    ]
    memmapstream(b, fmt)
    {'name': array([[65, 110, 100, 121]], dtype=uint8),
     'age': array([[5]], dtype=uint8),
     'school': array([[74, 84]], dtype=uint8)}
    """

    if not isinstance(bytes_in, (bytes, bytearray, np.ndarray)):
        raise TypeError("Input must be bytes, bytearray, or a uint8/int8 ndarray.")

    if isinstance(bytes_in, np.ndarray) and bytes_in.dtype not in [np.uint8, np.int8]:
        raise TypeError("NumPy input must be of dtype uint8 or int8.")

    if not (isinstance(format, list) and all(len(f) == 3 for f in format)):
        raise ValueError("Format must be a list of [dtype, shape, fieldname].")

    # flatten and ensure byte representation
    if isinstance(bytes_in, np.ndarray):
        byte_array = bytes_in.flatten(order="F").tobytes()
    else:
        byte_array = bytes(bytes_in)

    offset = 0
    outstruct = defaultdict()

    for dtype_str, shape, field in format:
        count = np.prod(shape)
        dtype_np = np.dtype(dtype_str)
        nbytes = count * dtype_np.itemsize

        if offset + nbytes > len(byte_array):
            break

        buffer = byte_array[offset : offset + nbytes]
        arr = np.frombuffer(buffer, dtype=dtype_np).reshape(shape, order="F")
        outstruct[field] = arr

        offset += nbytes

    return outstruct


def savejnii(*args, **kwargs):
    """
    Alias for jd.save
    """
    return jd.save(*args, **kwargs)


def savebnii(*args, **kwargs):
    """
    Alias for jd.save
    """
    return jd.save(*args, **kwargs)
