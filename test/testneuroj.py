"""neurojson.py test unit

To run the test, please run

   python3 -m unittest test.testneuroj

or

   import testneuroj
   testneuroj.run()

in the root folder.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.neurojson import neuroj

import numpy as np


class Test_neuroj(unittest.TestCase):
    def test_list_all(self):
        dat = neuroj("list")
        self.assertEqual(list(dat.keys()), ["_id", "_rev", "database", "dependency"])
        self.assertEqual(dat["_id"], "registry")
        self.assertGreater(len(dat["database"]), 20)
        dblist = [d["id"] for d in dat["database"]]
        self.assertTrue("openneuro" in dblist)
        self.assertTrue("mcx" in dblist)

    def test_list_db(self):
        dat = neuroj("list", db="mcx", limit=5, skip=5)
        self.assertEqual(list(dat.keys()), ["total_rows", "offset", "dataset"])
        self.assertEqual(dat["total_rows"], 20)
        self.assertEqual(dat["offset"], 5)
        dslist = [d["id"] for d in dat["dataset"]]
        self.assertTrue("colin27" in dslist)

    def test_list_dblimit(self):
        dat = neuroj("list", db="mcx", limit=5, skip=5)
        self.assertEqual(list(dat.keys()), ["total_rows", "offset", "dataset"])
        self.assertEqual(dat["total_rows"], 20)
        self.assertEqual(dat["offset"], 5)
        dslist = [d["id"] for d in dat["dataset"]]
        self.assertTrue("colin27" in dslist)

    def test_list_ds(self):
        dat = neuroj("list", db="mcx", ds="colin27")
        self.assertGreaterEqual(len(dat), 1)
        self.assertGreaterEqual(list(dat[0].keys()), ["rev", "status"])
        etags = dat[0]["rev"].split("-")
        self.assertGreater(int(etags[0]), 2)
        self.assertEqual(len(etags[1]), len("57c0527f2b210d5588f1b3dbec766e50"))

    def test_info_db(self):
        dat = neuroj("info", db="mcx")
        self.assertEqual(
            list(dat.keys()),
            [
                "instance_start_time",
                "db_name",
                "purge_seq",
                "update_seq",
                "sizes",
                "props",
                "doc_del_count",
                "doc_count",
                "disk_format_version",
                "compact_running",
                "cluster",
            ],
        )
        self.assertEqual(int(dat["instance_start_time"]), 1736744653)
        self.assertEqual(dat["db_name"], "mcx")
        self.assertGreater(dat["doc_count"], 10)

    def test_info_ds(self):
        dat = neuroj("info", db="mcx", ds="colin27")
        self.assertEqual(dat["Content-Type"], "application/json")
        self.assertGreater(int(dat["Content-Length"]), 50000)
        etags = dat["ETag"].replace('"', "").split("-")
        self.assertGreater(int(etags[0]), 2)
        self.assertEqual(len(etags[1]), len("57c0527f2b210d5588f1b3dbec766e50"))


if __name__ == "__main__":
    unittest.main()
