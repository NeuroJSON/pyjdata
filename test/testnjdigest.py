"""Tests for jdata.njdigest -- header-only readers for bulky containers.

These formats previously fell through the converter to an opaque link, so the
point of each test is twofold: that the searchable fields really are recovered,
and that the sample payload is *not* pulled into the digest.
"""

import os
import sys
import shutil
import struct
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jdata.njdigest import (
    edfheader,
    bvheader,
    hdf5_digest,
    snirf_digest,
    eeglab_digest,
    SNIRF_BULK,
)

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


def _ascii(value, width):
    return ("%-*s" % (width, value)).encode("latin-1")[:width]


def make_edf(path, nsignals=3, nrecords=10, recdur=1.0, samples=256, bdf=False):
    """Write a syntactically valid EDF/BDF file with deterministic content."""
    labels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"][:nsignals]
    head = bytearray()
    if bdf:
        head += b"\xff" + _ascii("BIOSEMI", 7)
    else:
        head += _ascii("0", 8)
    head += _ascii("MCH-0234567 F 02-MAY-1951 Haagse_Harry", 80)
    head += _ascii("Startdate 02-MAR-2002 PSG-1234/2002 NN Telemetry03", 80)
    head += _ascii("02.03.02", 8)
    head += _ascii("10.44.00", 8)
    head += _ascii(str(256 + 256 * nsignals), 8)
    head += _ascii("EDF+C" if not bdf else "24BIT", 44)
    head += _ascii(str(nrecords), 8)
    head += _ascii(("%g" % recdur), 8)
    head += _ascii(str(nsignals), 4)
    assert len(head) == 256, len(head)

    body = bytearray()
    for width, values in [
        (16, labels),
        (80, ["AgAgCl electrode"] * nsignals),
        (8, ["uV"] * nsignals),
        (8, ["-440"] * nsignals),
        (8, ["440"] * nsignals),
        (8, ["-2048"] * nsignals),
        (8, ["2047"] * nsignals),
        (80, ["HP:0.1Hz LP:75Hz"] * nsignals),
        (8, [str(samples)] * nsignals),
        (32, [""] * nsignals),
    ]:
        for value in values:
            body += _ascii(value, width)

    payload = b"\x00" * (nrecords * nsignals * samples * (3 if bdf else 2))
    with open(path, "wb") as fid:
        fid.write(bytes(head) + bytes(body) + payload)
    return path


VHDR = """Brain Vision Data Exchange Header File Version 1.0
; a comment line that must be ignored

[Common Infos]
Codepage=UTF-8
DataFile=sub-01_task-rest_eeg.eeg
MarkerFile=sub-01_task-rest_eeg.vmrk
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=4
SamplingInterval=2000

[Binary Infos]
BinaryFormat=INT_16

[Channel Infos]
Ch1=Fp1,,0.1,uV
Ch2=Fp2,,0.1,uV
Ch10=Cz,,0.5,uV
Ch3=O1,REF,0.1,uV

[Comment]
Amplifier setup line one
Amplifier setup line two
"""

VMRK = """Brain Vision Data Exchange Marker File, Version 1.0

[Common Infos]
DataFile=sub-01_task-rest_eeg.eeg

[Marker Infos]
Mk1=New Segment,,1,1,0,20200101120000000000
Mk2=Stimulus,S  1,1500,1,0
Mk3=Stimulus,S  2,3000,1,0
"""


class TestEdfHeader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_edf_fields(self):
        path = make_edf(os.path.join(self.root, "a.edf"))
        hdr = edfheader(path)["EDFHeader"]
        self.assertEqual(hdr["Format"], "EDF")
        self.assertEqual(hdr["NumberOfSignals"], 3)
        self.assertEqual(hdr["NumberOfDataRecords"], 10)
        self.assertEqual(hdr["DurationOfDataRecord"], 1.0)
        self.assertEqual(hdr["RecordingDuration"], 10.0)
        self.assertEqual(hdr["Labels"], ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"])
        self.assertEqual(hdr["PhysicalDimension"], ["uV", "uV", "uV"])
        self.assertEqual(hdr["PhysicalMinimum"], [-440.0, -440.0, -440.0])
        self.assertEqual(hdr["DigitalMaximum"], [2047, 2047, 2047])
        self.assertEqual(hdr["SamplesPerRecord"], [256, 256, 256])
        self.assertEqual(hdr["SamplingFrequency"], [256.0, 256.0, 256.0])
        self.assertEqual(hdr["StartDate"], "02.03.02")
        self.assertIn("EDF+C", hdr["Reserved"])

    def test_bdf_is_detected(self):
        path = make_edf(os.path.join(self.root, "b.bdf"), bdf=True)
        hdr = edfheader(path)["EDFHeader"]
        self.assertEqual(hdr["Format"], "BDF")
        self.assertEqual(hdr["Version"], "BIOSEMI")

    def test_fractional_record_duration(self):
        path = make_edf(os.path.join(self.root, "c.edf"), recdur=0.5, samples=128)
        hdr = edfheader(path)["EDFHeader"]
        self.assertEqual(hdr["SamplingFrequency"], [256.0] * 3)

    def test_reads_only_the_header(self):
        """A 10 MB recording must cost only 256 + 256*ns bytes of parsing."""
        path = make_edf(os.path.join(self.root, "d.edf"), nrecords=20000, samples=256)
        self.assertGreater(os.path.getsize(path), 10 << 20)
        hdr = edfheader(path)["EDFHeader"]
        # no sample data appears anywhere in the digest
        self.assertNotIn("Data", hdr)
        self.assertEqual(hdr["NumberOfDataRecords"], 20000)

    def test_truncated_file_raises(self):
        path = os.path.join(self.root, "trunc.edf")
        with open(path, "wb") as fid:
            fid.write(b"0" * 100)
        with self.assertRaises(ValueError):
            edfheader(path)


class TestBrainVisionHeader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.root, name)
        with open(path, "w") as fid:
            fid.write(text)
        return path

    def test_vhdr_common_infos(self):
        out = bvheader(self._write("a.vhdr", VHDR))["VHDRHeader"]
        common = out["CommonInfos"]
        self.assertEqual(common["NumberOfChannels"], 4)
        self.assertEqual(common["SamplingInterval"], 2000)
        self.assertEqual(common["DataFormat"], "BINARY")
        self.assertEqual(out["BinaryInfos"]["BinaryFormat"], "INT_16")

    def test_channels_are_column_oriented_and_numerically_ordered(self):
        out = bvheader(self._write("b.vhdr", VHDR))["VHDRHeader"]
        chans = out["ChannelInfos"]
        # Ch10 must sort after Ch3, not lexicographically between Ch1 and Ch2
        self.assertEqual(chans["name"], ["Fp1", "Fp2", "O1", "Cz"])
        self.assertEqual(chans["reference"], ["", "", "REF", ""])
        self.assertEqual(chans["resolution"], [0.1, 0.1, 0.1, 0.5])
        self.assertEqual(chans["unit"], ["uV", "uV", "uV", "uV"])

    def test_comment_block_is_preserved_as_text(self):
        out = bvheader(self._write("c.vhdr", VHDR))["VHDRHeader"]
        self.assertIn("Amplifier setup line one", out["Comment"])
        self.assertIn("Amplifier setup line two", out["Comment"])

    def test_vmrk_markers(self):
        out = bvheader(self._write("d.vmrk", VMRK))["VMRKHeader"]
        marks = out["MarkerInfos"]
        self.assertEqual(marks["type"], ["New Segment", "Stimulus", "Stimulus"])
        self.assertEqual(marks["description"], ["", "S  1", "S  2"])
        self.assertEqual(marks["position"], [1, 1500, 3000])

    def test_escaped_comma_in_channel_name(self):
        text = VHDR.replace("Ch1=Fp1,,0.1,uV", "Ch1=Fp\\11,,0.1,uV")
        out = bvheader(self._write("e.vhdr", text))["VHDRHeader"]
        self.assertEqual(out["ChannelInfos"]["name"][0], "Fp,1")


@unittest.skipUnless(HAVE_H5PY, "h5py is required")
class TestHdf5Digest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.path = os.path.join(self.root, "a.snirf")
        import numpy as np

        with h5py.File(self.path, "w") as fid:
            fid.create_dataset("formatVersion", data=b"1.1")
            nirs = fid.create_group("nirs")
            meta = nirs.create_group("metaDataTags")
            meta.create_dataset("SubjectID", data=b"sub-01")
            meta.create_dataset("MeasurementDate", data=b"2024-01-01")
            probe = nirs.create_group("probe")
            probe.create_dataset("wavelengths", data=np.array([760.0, 850.0]))
            probe.create_dataset("sourcePos3D", data=np.zeros((8, 3)))
            data1 = nirs.create_group("data1")
            # the bulk payload: small enough to slip under any element cap
            data1.create_dataset("dataTimeSeries", data=np.zeros((100, 4)))
            data1.create_dataset("time", data=np.arange(100.0))
            ml = data1.create_group("measurementList1")
            ml.create_dataset("sourceIndex", data=1)
            ml.create_dataset("wavelengthIndex", data=1)
            nirs.attrs["SNIRFversion"] = "1.1"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_metadata_is_inlined(self):
        digest = hdf5_digest(self.path, maxelem=256)
        nirs = digest["nirs"]
        self.assertEqual(nirs["metaDataTags"]["SubjectID"], "sub-01")
        self.assertEqual(nirs["probe"]["wavelengths"], [760.0, 850.0])
        self.assertEqual(
            nirs["measurementList1"]
            if "measurementList1" in nirs
            else nirs["data1"]["measurementList1"]["sourceIndex"],
            1,
        )

    def test_attributes_are_captured(self):
        digest = hdf5_digest(self.path, maxelem=256)
        self.assertEqual(digest["nirs"]["_Attributes_"]["SNIRFversion"], "1.1")

    def test_large_arrays_become_shape_annotations(self):
        digest = hdf5_digest(self.path, maxelem=8)
        pos = digest["nirs"]["probe"]["sourcePos3D"]
        self.assertEqual(pos["_ArraySize_"], [8, 3])
        self.assertEqual(pos["_ArrayType_"], "double")
        self.assertTrue(pos["_ArrayIsTruncated_"])
        self.assertNotIn("_ArrayData_", pos)

    def test_snirf_drops_sample_data_by_name(self):
        """dataTimeSeries/time must be excluded even when small enough to inline."""
        plain = hdf5_digest(self.path, maxelem=100000)["nirs"]["data1"]
        self.assertIsInstance(plain["time"], list)  # would have been inlined

        digest = snirf_digest(self.path, maxelem=100000)["SNIRFData"]["nirs"]["data1"]
        self.assertTrue(digest["dataTimeSeries"]["_ArrayIsTruncated_"])
        self.assertEqual(digest["dataTimeSeries"]["_ArraySize_"], [100, 4])
        self.assertTrue(digest["time"]["_ArrayIsTruncated_"])
        # while the descriptive parts survive
        self.assertEqual(digest["measurementList1"]["sourceIndex"], 1)

    def test_snirf_bulk_pattern_is_anchored(self):
        import re

        pattern = re.compile(SNIRF_BULK)
        self.assertTrue(pattern.search("dataTimeSeries"))
        self.assertTrue(pattern.search("time"))
        self.assertFalse(pattern.search("timeOffsetSomethingElse"))
        self.assertFalse(pattern.search("measurementList1"))

    def test_eeglab_digest_routes_hdf5(self):
        out = eeglab_digest(self.path, maxelem=8)
        self.assertIn("EEGLABData", out)


if __name__ == "__main__":
    unittest.main()


class TestMatDetection(unittest.TestCase):
    """A .mat extension does not guarantee a MATLAB file."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, data, mode="wb"):
        path = os.path.join(self.root, name)
        with open(path, mode) as fid:
            fid.write(data)
        return path

    def test_matv5_is_detected(self):
        from jdata.njdigest import is_matfile

        body = b"MATLAB 5.0 MAT-file, created by test".ljust(124, b" ") + b"\x00\x01" + b"IM"
        self.assertTrue(is_matfile(self._write("a.mat", body)))

    def test_hdf5_matv73_is_detected(self):
        from jdata.njdigest import is_matfile

        self.assertTrue(is_matfile(self._write("b.mat", b"\x89HDF\r\n\x1a\n" + b"\x00" * 200)))

    def test_fsl_vest_matrix_is_not_a_matfile(self):
        from jdata.njdigest import is_matfile

        text = "/NumWaves 5\n/NumPoints 26\n/Matrix\n1 2 3 4 5\n"
        self.assertFalse(is_matfile(self._write("c.mat", text, mode="w")))

    def test_vest_header_fields(self):
        from jdata.njdigest import vest_header

        text = (
            "/NumWaves 3\n/NumPoints 4\n/PPheights 1 1 1\n/Matrix\n"
            "1\t2\t3\n4\t5\t6\n7\t8\t9\n10\t11\t12\n"
        )
        header = vest_header(self._write("d.mat", text, mode="w"))["VESTHeader"]
        self.assertEqual(header["NumWaves"], 3)
        self.assertEqual(header["NumPoints"], 4)
        self.assertEqual(header["PPheights"], [1, 1, 1])
        self.assertEqual(header["NumRows"], 4)

    def test_vest_header_returns_none_for_unrelated_text(self):
        from jdata.njdigest import vest_header

        blob = "\n".join("this is not a design matrix" for _ in range(200))
        self.assertIsNone(vest_header(self._write("e.mat", blob, mode="w")))

    def test_vest_reads_only_the_header_not_the_matrix(self):
        from jdata.njdigest import vest_header

        rows = "\n".join("\t".join("%.6f" % (i + j) for j in range(20)) for i in range(50000))
        path = self._write("f.mat", "/NumWaves 20\n/Matrix\n" + rows + "\n", mode="w")
        self.assertGreater(os.path.getsize(path), 5 << 20)
        header = vest_header(path)["VESTHeader"]
        self.assertEqual(header["NumWaves"], 20)
        self.assertNotIn("Matrix", header)
