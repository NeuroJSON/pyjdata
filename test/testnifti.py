"""jnifti.py test unit

To run the test, please run

   python3 -m unittest test.testnifti

or

   import testnifti
   testnifti.run()

in the root folder.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import struct
import shutil
import tempfile
import unittest

try:
    import nibabel  # noqa: F401

    HAVE_NIBABEL = True
except ImportError:
    HAVE_NIBABEL = False
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.jnifti import savenifti, nii2jnii, jnii2nii, niiheader2jnii, nifticreate
from jdata.jfile import loadurl

import numpy as np


class Test_jnifti(unittest.TestCase):
    @classmethod
    def setUpClass(self, *args, **kwargs):
        self.jnii = loadurl("https://neurojson.io:7777/unc-012-infant-atlas/infant-1yr-seg")
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
        self.assertEqual(hdr["Affine"], [[1, 0, 0, -89], [0, 1, 0, -125], [0, 0, 1, -71]])

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
        self.assertEqual(hdr["Affine"].tolist(), [[1, 0, 0, -89], [0, 1, 0, -125], [0, 0, 1, -71]])

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


class TestNiftiExtensions(unittest.TestCase):
    """A NIfTI carrying an extension record used to fail to parse at all.

    Three separate defects stacked up in the extension block, and every one of
    them was masked by the fact that most files have no extensions and so never
    reach it:

      * the header fields are 1-element numpy arrays, and using them raw as a
        slice index raises TypeError;
      * dataendian held "little"/"big" from sys.byteorder but was handed to
        struct as a format prefix, where only "<" and ">" are valid;
      * the same variable was compared against "L", which sys.byteorder never
        returns, so nii["endian"] reported big-endian for every file ever read.
    """

    def _write_with_extension(self, path, endian="<"):
        """A minimal NIfTI-1 with one 16-byte extension record."""
        e = endian
        hdr = bytearray(348)
        hdr[0:4] = struct.pack(e + "i", 348)
        hdr[40:42] = struct.pack(e + "h", 3)          # dim[0] = 3
        hdr[42:48] = struct.pack(e + "3h", 2, 3, 4)   # 2x3x4
        hdr[70:72] = struct.pack(e + "h", 16)         # datatype float32
        hdr[72:74] = struct.pack(e + "h", 32)         # bitpix
        hdr[76:80] = struct.pack(e + "f", 1.0)        # pixdim[0]
        hdr[80:92] = struct.pack(e + "3f", 1.0, 1.0, 1.0)
        voxoffset = 348 + 4 + 16
        hdr[108:112] = struct.pack(e + "f", float(voxoffset))
        hdr[344:348] = b"n+1\x00"
        ext = struct.pack(e + "4B", 1, 0, 0, 0)       # extension flag
        ext += struct.pack(e + "ii", 16, 4) + b"hello ext"[:8]
        vol = np.arange(24, dtype=e + "f4").reshape(2, 3, 4, order="F")
        with open(path, "wb") as fid:
            fid.write(bytes(hdr) + ext + vol.tobytes(order="F"))

    def test_extension_record_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "ext.nii")
            self._write_with_extension(f)
            nii = nii2jnii(f)                    # used to raise TypeError
            self.assertIn("NIFTIExtension", nii)
            self.assertEqual(len(nii["NIFTIExtension"]), 1)
            self.assertEqual(nii["NIFTIExtension"][0]["Type"], 4)
            self.assertEqual(
                np.asarray(nii["NIFTIData"]).shape, (2, 3, 4)
            )

    def test_big_endian_extension_parses(self):
        """Exercises the byte-order path: the struct prefix used to be the word
        "little", which is not a valid format character."""
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "be.nii")
            self._write_with_extension(f, endian=">")
            nii = nii2jnii(f)
            self.assertEqual(len(nii["NIFTIExtension"]), 1)
            self.assertEqual(nii["NIFTIExtension"][0]["Type"], 4)



if __name__ == "__main__":
    unittest.main()


class TestVoxelAxisOrder(unittest.TestCase):
    """NIfTI stores voxels in column-major order.

    Regression test.  NumPy's ``reshape`` defaults to row-major, so the Python
    port disagreed with the MATLAB reference (``nii2jnii.m`` uses MATLAB's
    ``reshape``, which is column-major) and with every other NIfTI reader.  It
    went unnoticed because the write path made the matching assumption, so jdata
    round-tripped with itself perfectly -- while writing files that other tools
    read wrongly, and reading theirs wrongly in turn.

    The first test needs no external library: it builds a byte buffer whose
    correct interpretation is known from the format alone.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_nifti1(self, shape, values, dtype=np.int16, datatype=4, bitpix=16):
        """Write a NIfTI-1 file with a payload laid out column-major."""
        hdr = bytearray(348)
        struct.pack_into("<i", hdr, 0, 348)
        struct.pack_into("<h", hdr, 40, len(shape))
        for index, dim in enumerate(shape):
            struct.pack_into("<h", hdr, 42 + 2 * index, dim)
        struct.pack_into("<h", hdr, 70, datatype)
        struct.pack_into("<h", hdr, 72, bitpix)
        struct.pack_into("<f", hdr, 76, 1.0)
        for index in range(len(shape)):
            struct.pack_into("<f", hdr, 80 + 4 * index, 1.0)
        struct.pack_into("<f", hdr, 108, 352.0)
        struct.pack_into("<f", hdr, 112, 1.0)
        hdr[344:348] = b"n+1\x00"
        path = os.path.join(self.root, "case.nii")
        with open(path, "wb") as fid:
            fid.write(bytes(hdr) + b"\x00" * 4)
            fid.write(np.asarray(values, dtype=dtype).tobytes(order="F"))
        return path

    def test_column_major_payload_is_read_back_correctly(self):
        """Known layout, no external reader involved."""
        expected = np.arange(2 * 3 * 4, dtype=np.int16).reshape((2, 3, 4))
        path = self._write_nifti1((2, 3, 4), expected)
        got = np.asarray(nii2jnii(path)["NIFTIData"])
        self.assertEqual(got.shape, (2, 3, 4))
        self.assertTrue(np.array_equal(got, expected))

    def test_first_voxels_follow_the_fastest_varying_axis(self):
        """Element order in the file is x fastest, so [0,0,0] then [1,0,0]."""
        expected = np.arange(3 * 4 * 5, dtype=np.int16).reshape((3, 4, 5))
        path = self._write_nifti1((3, 4, 5), expected)
        got = np.asarray(nii2jnii(path)["NIFTIData"])
        self.assertEqual(int(got[0, 0, 0]), int(expected[0, 0, 0]))
        self.assertEqual(int(got[1, 0, 0]), int(expected[1, 0, 0]))
        self.assertEqual(int(got[0, 1, 0]), int(expected[0, 1, 0]))

    def test_non_cubic_shape_would_fail_under_row_major(self):
        """A shape with distinct extents cannot survive the wrong order."""
        expected = np.arange(2 * 7 * 3, dtype=np.int16).reshape((2, 7, 3))
        path = self._write_nifti1((2, 7, 3), expected)
        got = np.asarray(nii2jnii(path)["NIFTIData"])
        self.assertTrue(np.array_equal(got, expected))
        self.assertFalse(np.array_equal(got, expected.ravel().reshape((2, 7, 3))[::-1]))

    def test_four_dimensional_volume(self):
        expected = np.arange(3 * 4 * 5 * 6, dtype=np.int16).reshape((3, 4, 5, 6))
        path = self._write_nifti1((3, 4, 5, 6), expected)
        self.assertTrue(np.array_equal(np.asarray(nii2jnii(path)["NIFTIData"]), expected))

    def test_write_then_read_round_trip(self):
        for shape in ((2, 3, 4), (5, 6, 7, 8), (11, 13)):
            data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
            path = os.path.join(self.root, "rt%d.nii" % len(shape))
            savenifti(data, path)
            self.assertTrue(
                np.array_equal(np.asarray(nii2jnii(path)["NIFTIData"]), data), str(shape)
            )

    @unittest.skipUnless(HAVE_NIBABEL, "nibabel is required for the cross-check")
    def test_agrees_with_nibabel_on_read(self):
        expected = np.arange(3 * 4 * 5, dtype=np.int16).reshape((3, 4, 5))
        path = self._write_nifti1((3, 4, 5), expected)
        import nibabel

        self.assertTrue(
            np.array_equal(
                np.asarray(nii2jnii(path)["NIFTIData"]),
                np.asanyarray(nibabel.load(path).dataobj),
            )
        )

    @unittest.skipUnless(HAVE_NIBABEL, "nibabel is required for the cross-check")
    def test_files_written_by_jdata_are_read_correctly_by_nibabel(self):
        """The part that mattered: our output must be right for other tools."""
        import nibabel

        for shape in ((2, 3, 4), (5, 6, 7, 8)):
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            path = os.path.join(self.root, "out%d.nii" % len(shape))
            savenifti(data, path)
            self.assertTrue(
                np.array_equal(np.asanyarray(nibabel.load(path).dataobj), data), str(shape)
            )
