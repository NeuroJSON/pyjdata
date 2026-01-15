"""jnifti.py test unit

To run the test, please run

   python3 -m unittest test.testnifti

or

   import testnifti
   testnifti.run()

in the root folder.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.jnifti import jnii2nii, niiheader2jnii, nifticreate
from jdata.jfile import loadurl

import numpy as np


class Test_jnifti(unittest.TestCase):
    @classmethod
    def setUpClass(self, *args, **kwargs):
        self.jnii = loadurl(
            "https://neurojson.io:7777/unc-012-infant-atlas/infant-1yr-seg"
        )
        self.nii = jnii2nii(self.jnii)

    def test_jnii_img(self):
        hdr = self.jnii["NIFTIHeader"]
        img = self.jnii["NIFTIData"]
        self.assertEqual(img.shape, (181, 217, 181))
        self.assertEqual(img.dtype.name, "uint16")
        self.assertEqual(img.shape, tuple(hdr["Dim"]))

    def test_jnii_hdr(self):
        hdr = self.jnii["NIFTIHeader"]
        self.assertEqual(hdr["DataType"], "int16")
        self.assertEqual(hdr["BitDepth"], 16)
        self.assertEqual(hdr["NIIHeaderSize"], 348)
        self.assertEqual(hdr["NIIByteOffset"], 352)
        self.assertEqual(hdr["NIIFormat"], "n+1")
        self.assertEqual(hdr["NIIQfac_"], 1)
        self.assertEqual(hdr["NIIEndian_"], "L")
        self.assertEqual(hdr["SForm"], 1)
        self.assertEqual(hdr["VoxelSize"], [1, 1, 1])
        self.assertEqual(hdr["NIIExtender"], [0, 0, 0, 0])
        self.assertEqual(hdr["Quatern"]["d"], 1)
        self.assertEqual(
            hdr["Affine"], [[1, 0, 0, -89], [0, 1, 0, -125], [0, 0, 1, -71]]
        )

    def test_jnii2nii(self):
        hdr = self.nii["hdr"]
        img = self.nii["img"]
        self.assertEqual(img.shape, (181, 217, 181))
        self.assertEqual(img.dtype.name, "uint16")
        self.assertEqual(img.shape, tuple(hdr["dim"][1:4].tolist()))
        self.assertEqual(hdr["dim"].tolist(), [3, 181, 217, 181, 1, 1, 1, 1])
        self.assertEqual(hdr["dim"].dtype.name, "uint16")
        self.assertEqual(hdr["regular"], 114)
        self.assertEqual(hdr["regular"].dtype.name, "int8")
        self.assertEqual(hdr["sizeof_hdr"], 348)
        self.assertEqual(hdr["sizeof_hdr"].dtype.name, "int32")
        self.assertEqual(hdr["glmax"], 250)
        self.assertEqual(hdr["glmax"].dtype.name, "int32")
        self.assertEqual(hdr["xyzt_units"], 2)
        self.assertEqual(hdr["xyzt_units"].dtype.name, "int8")
        self.assertEqual(hdr["vox_offset"], 352)
        self.assertEqual(hdr["vox_offset"].dtype.name, "float32")
        self.assertEqual(hdr["scl_slope"], 1)
        self.assertEqual(hdr["scl_slope"].dtype.name, "float32")
        self.assertEqual(hdr["pixdim"].tolist(), [1] * 8)
        self.assertEqual(hdr["datatype"], 4)
        self.assertEqual(hdr["datatype"].dtype.name, "int16")
        self.assertEqual(hdr["bitpix"], 16)
        self.assertEqual(hdr["srow_x"].tolist(), [1.0, 0.0, 0.0, -89.0])
        self.assertEqual(hdr["srow_x"].dtype.name, "float32")
        self.assertEqual(hdr["srow_z"].tolist(), [0.0, 0.0, 1.0, -71.0])
        self.assertEqual(hdr["srow_z"].dtype.name, "float32")
        self.assertEqual(bytearray(hdr["magic"]), b"n+1\x00")

    def test_nii_buffer(self):
        buf = b"".join(self.nii["hdr"][name].tobytes() for name in self.nii["hdr"])
        self.assertEqual(len(buf), 352)

    def test_niiheader2jnii(self):
        jnii = niiheader2jnii(self.nii)
        hdr = jnii["NIFTIHeader"]
        self.assertEqual(hdr["DataType"], "int16")
        self.assertEqual(hdr["BitDepth"], 16)
        self.assertEqual(hdr["NIIHeaderSize"], 348)
        self.assertEqual(hdr["NIIByteOffset"], 352)
        self.assertEqual(hdr["NIIFormat"], "n+1")
        self.assertEqual(hdr["NIIQfac_"], 1)
        self.assertEqual(hdr["NIIEndian_"], "little")
        self.assertEqual(hdr["SForm"], 1)
        self.assertEqual(hdr["VoxelSize"].tolist(), [1, 1, 1])
        self.assertEqual(hdr["NIIExtender"].tolist(), [0, 0, 0, 0])
        self.assertEqual(hdr["Quatern"]["d"], 1)
        self.assertEqual(
            hdr["Affine"].tolist(), [[1, 0, 0, -89], [0, 1, 0, -125], [0, 0, 1, -71]]
        )

    def test_nifticreate_nifti1(self):
        nii = nifticreate(np.ones((4, 5, 6), dtype=np.float32))
        hdr = nii["hdr"]
        img = nii["img"]

        buf = b"".join(nii["hdr"][name].tobytes() for name in nii["hdr"])
        self.assertEqual(len(buf), 352)

        self.assertEqual(hdr["dim"].tolist(), [3, 4, 5, 6, 1, 1, 1, 1])
        self.assertEqual(hdr["dim"].dtype.name, "uint16")
        self.assertEqual(img.shape, tuple(hdr["dim"][1:4].tolist()))
        self.assertEqual(img.dtype.name, "float32")
        self.assertEqual(np.sum(img), 4 * 5 * 6)
        self.assertEqual(hdr["sizeof_hdr"], 348)
        self.assertEqual(hdr["srow_x"].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(hdr["srow_z"].tolist(), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(hdr["datatype"], 16)

    def test_nifticreate_nifti2(self):
        nii = nifticreate(np.ones((4, 5, 6), dtype=np.int32), "nifti2")
        hdr = nii["hdr"]
        img = nii["img"]

        buf = b"".join(nii["hdr"][name].tobytes() for name in nii["hdr"])
        self.assertEqual(len(buf), 544)

        self.assertEqual(img.shape, tuple(hdr["dim"][1:4].tolist()))
        self.assertEqual(img.dtype.name, "int32")

        self.assertEqual(hdr["sizeof_hdr"], 540)
        self.assertEqual(hdr["dim"].tolist(), [3, 4, 5, 6, 1, 1, 1, 1])
        self.assertEqual(hdr["dim"].dtype.name, "int64")

        self.assertEqual(np.sum(img), np.int32(4 * 5 * 6))
        self.assertEqual(bytearray(hdr["magic"]), b"ni2\x00\x00\x00\x00\x00")
        self.assertEqual(hdr["datatype"], 8)
        self.assertEqual(hdr["datatype"].dtype.name, "int16")
        self.assertEqual(hdr["pixdim"].dtype.name, "float64")

    def test_nifticreate_convert_nifti2(self):
        nii = nifticreate(self.nii, "nifti2")
        hdr = nii["hdr"]
        img = nii["img"]
        self.assertEqual(img.shape, (181, 217, 181))
        self.assertEqual(img.dtype.name, "uint16")

        buf = b"".join(nii["hdr"][name].tobytes() for name in nii["hdr"])
        self.assertEqual(len(buf), 544)

        self.assertEqual(img.shape, tuple(hdr["dim"][1:4].tolist()))
        self.assertEqual(img.dtype.name, "uint16")

        self.assertEqual(hdr["sizeof_hdr"], 540)
        self.assertEqual(hdr["dim"].tolist(), [3, 181, 217, 181, 1, 1, 1, 1])
        self.assertEqual(hdr["dim"].dtype.name, "int64")

        self.assertEqual(np.sum(img), 146787100)
        self.assertEqual(bytearray(hdr["magic"]), b"ni2\x00\x00\x00\x00\x00")
        self.assertEqual(hdr["datatype"], 512)
        self.assertEqual(hdr["datatype"].dtype.name, "int16")
        self.assertEqual(hdr["pixdim"].dtype.name, "float64")


if __name__ == "__main__":
    unittest.main()
