"""h5.py SNIRF/HDF5 test unit

To run the test, please run

   python3 -m unittest test.testsnirf

or

   import testsnirf
   testsnirf.run()

in the root folder.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.h5 import regrouph5, loadh5, saveh5
from jdata.jfile import loadjd, loadsnirf, loadjsnirf, savesnirf, savejsnirf

import numpy as np


class Test_snirf(unittest.TestCase):
    @classmethod
    def setUpClass(self, *args, **kwargs):
        self.snirfdata = loadjd(
            "https://github.com/fNIRS/snirf-samples/raw/refs/heads/master/basic/Simple_Probe.snirf"
        )

    def test_snirf(self):
        self.assertEqual(list(self.snirfdata.keys()), ["formatVersion", "nirs"])
        self.assertEqual(self.snirfdata["formatVersion"], "1.0")
        self.assertEqual(
            list(self.snirfdata["nirs"].keys()),
            ["metaDataTags", "data", "probe", "stim", "aux"],
        )
        self.assertEqual(len(self.snirfdata["nirs"]["data"]), 1)
        self.assertEqual(
            list(self.snirfdata["nirs"]["data"][0].keys()),
            ["dataTimeSeries", "measurementList", "time"],
        )
        datashape = self.snirfdata["nirs"]["data"][0]["dataTimeSeries"].shape
        self.assertEqual(datashape, (1200, 8))
        self.assertEqual(
            self.snirfdata["nirs"]["data"][0]["time"].shape, (datashape[0],)
        )
        self.assertEqual(
            len(self.snirfdata["nirs"]["data"][0]["measurementList"]), datashape[1]
        )
        self.assertEqual(
            list(self.snirfdata["nirs"]["data"][0]["measurementList"][0].keys()),
            [
                "dataType",
                "dataTypeIndex",
                "detectorGain",
                "detectorIndex",
                "moduleIndex",
                "sourceIndex",
                "sourcePower",
                "wavelengthIndex",
            ],
        )

    def test_regrouph5(self):
        a = {
            "a1": np.random.rand(5, 4),
            "a2": "string",
            "a3": True,
            "d": 2 + 3j,
            "e": ["test", None, list(range(1, 6))],
        }
        b = regrouph5(a)
        self.assertEqual(len(b["a"]), 3)
        self.assertEqual(b["a"][0].tolist(), a["a1"].tolist())
        self.assertEqual(b["a"][1], a["a2"])
        self.assertEqual(b["a"][2], a["a3"])

    def test_snirf(self):
        self.assertEqual(list(self.snirfdata.keys()), ["formatVersion", "nirs"])
        self.assertEqual(self.snirfdata["formatVersion"], "1.0")
        self.assertEqual(
            list(self.snirfdata["nirs"].keys()),
            ["metaDataTags", "data", "probe", "stim", "aux"],
        )
        self.assertEqual(len(self.snirfdata["nirs"]["data"]), 1)
        self.assertEqual(
            list(self.snirfdata["nirs"]["data"][0].keys()),
            ["dataTimeSeries", "measurementList", "time"],
        )
        datashape = self.snirfdata["nirs"]["data"][0]["dataTimeSeries"].shape
        self.assertEqual(datashape, (1200, 8))
        self.assertEqual(
            self.snirfdata["nirs"]["data"][0]["time"].shape, (datashape[0],)
        )
        self.assertEqual(
            len(self.snirfdata["nirs"]["data"][0]["measurementList"]), datashape[1]
        )
        self.assertEqual(
            list(self.snirfdata["nirs"]["data"][0]["measurementList"][0].keys()),
            [
                "dataType",
                "dataTypeIndex",
                "detectorGain",
                "detectorIndex",
                "moduleIndex",
                "sourceIndex",
                "sourcePower",
                "wavelengthIndex",
            ],
        )

    def test_h5(self):
        h5 = loadjd(
            "https://github.com/fNIRS/snirf-samples/raw/refs/heads/master/basic/Simple_Probe.snirf",
            ".h5",
        )[0]
        self.assertEqual(list(h5.keys()), ["formatVersion", "nirs"])
        self.assertEqual(h5["formatVersion"], "1.0")
        self.assertEqual(
            list(h5["nirs"].keys()),
            ["metaDataTags", "data1", "probe", "stim1", "stim2", "stim3", "aux1"],
        )
        self.assertEqual(len(h5["nirs"]["data1"]), 10)
        self.assertEqual(
            list(h5["nirs"]["data1"].keys()),
            [
                "dataTimeSeries",
                "measurementList1",
                "measurementList2",
                "measurementList3",
                "measurementList4",
                "measurementList5",
                "measurementList6",
                "measurementList7",
                "measurementList8",
                "time",
            ],
        )
        datashape = h5["nirs"]["data1"]["dataTimeSeries"].shape
        self.assertEqual(datashape, (1200, 8))
        self.assertEqual(h5["nirs"]["data1"]["time"].shape, (datashape[0],))
        self.assertEqual(len(h5["nirs"]["data1"]["measurementList1"]), datashape[1])
        self.assertEqual(
            list(h5["nirs"]["data1"]["measurementList1"].keys()),
            [
                "dataType",
                "dataTypeIndex",
                "detectorGain",
                "detectorIndex",
                "moduleIndex",
                "sourceIndex",
                "sourcePower",
                "wavelengthIndex",
            ],
        )


if __name__ == "__main__":
    unittest.main()
