"""jdata.py test unit

To run the test, please run

   python3 -m unittest discover -v test

or

   import testjd
   testjd.run()

in the root folder.

Copyright (c) 2019-2024 Qianqian Fang <q.fang at neu.edu>
"""

import unittest

from jdata import *
import numpy as np
import re
import warnings
from collections import OrderedDict


def debug_print(data, opt={}):
    opt.setdefault("string", True)
    return show(data, opt, separators=(",", ":"))


def test_jdata(testname, fhandle, input, expected, *varargs):
    res = fhandle(input, *varargs)
    if not (str(res).strip() == expected):
        warnings.warn(
            f'Test {testname}: failed: expected "{expected}", obtained "{res}"'
        )
    else:
        print(f'Testing {testname}: ok\n\toutput:"{str(res).strip()}"')


class TestModule(unittest.TestCase):
    def test_json(self):
        print("\n")
        print("".join(["=" for _ in range(79)]))
        print("Test JSON functions")
        print("".join(["=" for _ in range(79)]))

        test_jdata("single integer", debug_print, 5, "5")
        test_jdata("single float", debug_print, 3.14, "3.14")
        test_jdata("nan", debug_print, float("nan"), '"_NaN_"')
        test_jdata("inf", debug_print, float("inf"), '"_Inf_"')
        test_jdata("-inf", debug_print, float("-inf"), '"-_Inf_"')
        test_jdata("large integer", debug_print, 2**64, "18446744073709551616")
        test_jdata(
            "large negative integer", debug_print, -(2**63), "-9223372036854775808"
        )
        test_jdata("boolean as 01", debug_print, [True, False], "[true,false]")
        test_jdata("empty array", debug_print, [], "[]")
        test_jdata("empty cell", debug_print, [], "[]")
        test_jdata("empty struct", debug_print, {}, "{}")
        test_jdata("empty struct with fields", debug_print, [], "[]")
        test_jdata("empty string", debug_print, "", '""')
        test_jdata(
            "string escape",
            debug_print,
            'jdata\n\b\ashall\tprevail\t""\\',
            '"jdata\\n\\b\\u0007shall\\tprevail\\t\\"\\"\\\\"',
        )
        test_jdata(
            "string type",
            debug_print,
            "jdata\n\b\ashall\tprevail",
            '"jdata\\n\\b\\u0007shall\\tprevail"',
        )
        test_jdata(
            "string array",
            debug_print,
            ["jdata", "shall", "prevail"],
            '["jdata","shall","prevail"]',
        )
        test_jdata("empty name", debug_print, loadts('{"":""}'), '{"":""}')
        test_jdata(
            "empty name with map",
            debug_print,
            loadts('{"":""}', {"usemap": 1}),
            '{"":""}',
        )
        test_jdata("row vector", debug_print, [1, 2, 3], "[1,2,3]")
        test_jdata("column vector", debug_print, [[1], [2], [3]], "[[1],[2],[3]]")
        test_jdata("mixed array", debug_print, ["a", 1, 0.9], '["a",1,0.9]')
        test_jdata(
            "mixed array from string",
            debug_print,
            loadts('["a",{"c":1}, [2,3]]'),
            '["a",{"c":1},[2,3]]',
        )
        test_jdata("char array", debug_print, ["AC", "EG"], '["AC","EG"]')
        test_jdata("maps", debug_print, {"a": 1, "b": "test"}, '{"a":1,"b":"test"}')
        test_jdata("2d array", debug_print, [[1, 2, 3], [4, 5, 6]], "[[1,2,3],[4,5,6]]")
        test_jdata(
            "non-uniform 2d array",
            debug_print,
            [[1, 2], [3, 4, 5], [6, 7]],
            "[[1,2],[3,4,5],[6,7]]",
        )
        test_jdata(
            "non-uniform array with length multiple of first element",
            debug_print,
            [[1, 2], [3, 4, 5, 6], [7, 8]],
            "[[1,2],[3,4,5,6],[7,8]]",
        )

        a = np.zeros((20, 5), np.uint8)
        np.fill_diagonal(a, 1)
        a[19, 0] = 1

        test_jdata(
            "zlib/zip compression (level 6)",
            debug_print,
            a,
            '{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipType_":"zlib","_ArrayZipSize_":[1,100],"_ArrayZipData_":"eJxjZAABRhwkxQBsDAACIQAH"}',
            {"compact": 1, "compression": "zlib", "compressarraysize": 0},
        )
        test_jdata(
            "gzip compression (level 6)",
            debug_print,
            a,
            '{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipType_":"gzip","_ArrayZipSize_":[1,100],"_ArrayZipData_":"H4sIAAAAAAAAAw=="}',
            {"compact": 1, "compression": "gzip", "compressarraysize": 0},
        )
        test_jdata(
            "lzma compression (level 5)",
            debug_print,
            a,
            '{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipType_":"lzma","_ArrayZipSize_":[1,100],"_ArrayZipData_":"XQAAgAD//////////wAAgD1IirvlZSEY7DH///taoAA="}',
            {"compact": 1, "compression": "lzma", "compressarraysize": 0},
        )

    def test_jsonpath(self):
        print("\n")
        print("".join(["=" for _ in range(79)]))
        print("Test JSONPath")
        print("".join(["=" for _ in range(79)]))

        testdata = {
            "book": [
                {"title": "Minch", "author": "Yoda"},
                {"title": "Qui-Gon", "author": "Jinn"},
                {"title": "Ben", "author": "Kenobi"},
            ],
            "game": {"title": "Mario", "new": {"title": "Minecraft"}},
        }
        test_jdata(
            "jsonpath of .key",
            debug_print,
            jsonpath(testdata, "$.game.title"),
            '"Mario"',
        )
        test_jdata(
            "jsonpath of ..key",
            debug_print,
            jsonpath(testdata, "$.book..title"),
            '["Minch","Qui-Gon","Ben"]',
        )
        test_jdata(
            "jsonpath of ..key cross objects",
            debug_print,
            jsonpath(testdata, "$..title"),
            '["Minch","Qui-Gon","Ben","Mario","Minecraft"]',
        )
        test_jdata(
            "jsonpath of [index]",
            debug_print,
            jsonpath(testdata, "$..title[1]"),
            '["Qui-Gon"]',
        )
        test_jdata(
            "jsonpath of [-index]",
            debug_print,
            jsonpath(testdata, "$..title[-1]"),
            '["Minecraft"]',
        )
        test_jdata(
            "jsonpath of [start:end]",
            debug_print,
            jsonpath(testdata, "$..title[0:2]"),
            '["Minch","Qui-Gon","Ben"]',
        )
        test_jdata(
            "jsonpath of [:end]",
            debug_print,
            jsonpath(testdata, "$..title[:2]"),
            '["Minch","Qui-Gon","Ben"]',
        )
        test_jdata(
            "jsonpath of [start:]",
            debug_print,
            jsonpath(testdata, "$..title[1:]"),
            '["Qui-Gon","Ben","Mario","Minecraft"]',
        )
        test_jdata(
            "jsonpath of [-start:-end]",
            debug_print,
            jsonpath(testdata, "$..title[-2:-1]"),
            '["Mario","Minecraft"]',
        )
        test_jdata(
            "jsonpath of [-start:]",
            debug_print,
            jsonpath(testdata, "$..title[:-3]"),
            '["Minch","Qui-Gon","Ben"]',
        )
        test_jdata(
            "jsonpath of [:-end]",
            debug_print,
            jsonpath(testdata, "$..title[-1:]"),
            '["Minecraft"]',
        )
        test_jdata(
            "jsonpath of object with [index]",
            debug_print,
            jsonpath(testdata, "$.book[1]"),
            '[{"title":"Qui-Gon","author":"Jinn"}]',
        )
        test_jdata(
            "jsonpath of element after [index]",
            debug_print,
            jsonpath(testdata, "$.book[1:2].author"),
            '["Jinn","Kenobi"]',
        )
        test_jdata(
            "jsonpath of [*] and deep scan",
            debug_print,
            jsonpath(testdata, "$.book[*]..author"),
            '["Yoda","Jinn","Kenobi"]',
        )
        test_jdata(
            "jsonpath of [*] after deep scan",
            debug_print,
            jsonpath(testdata, "$.book[*]..author[*]"),
            '["Yoda","Jinn","Kenobi"]',
        )
        test_jdata(
            "jsonpath use [] instead of .",
            debug_print,
            jsonpath(testdata, "$[book][2][author]"),
            '"Kenobi"',
        )
        test_jdata(
            "jsonpath use [] with [start:end]",
            debug_print,
            jsonpath(testdata, "$[book][1:2][author]"),
            '["Jinn","Kenobi"]',
        )
        test_jdata(
            "jsonpath use . after [start:end]",
            debug_print,
            jsonpath(testdata, "$[book][0:1].author"),
            '["Yoda","Jinn"]',
        )
        test_jdata(
            "jsonpath use [" "*" '] and ["*"]',
            debug_print,
            jsonpath(testdata, '$["book"][:-2][' "author" "]"),
            '["Yoda","Jinn"]',
        )
        test_jdata(
            "jsonpath use combinations",
            debug_print,
            jsonpath(testdata, '$..["book"][:-2].author[*][0]'),
            '["Yoda"]',
        )

        testdata = {
            "book": [
                {"_title": "Minch", " author.last.name ": "Yoda"},
                {"_title": "Qui-Gon", " author.last.name ": "Jinn"},
                {"_title": "Ben", " author.last.name ": "Kenobi"},
            ],
            "game.arcade": {"title": "Mario"},
        }
        test_jdata(
            "jsonpath encoded field name in []",
            debug_print,
            jsonpath(testdata, '$..["book"][_title][*][0]'),
            '["Minch"]',
        )
        test_jdata(
            "jsonpath encoded field name after .",
            debug_print,
            jsonpath(testdata, '$..["book"]._title[*][0]'),
            '["Minch"]',
        )
        test_jdata(
            "jsonpath encoded field name after ..",
            debug_print,
            jsonpath(testdata, "$.._title"),
            '["Minch","Qui-Gon","Ben"]',
        )
        test_jdata(
            "jsonpath multiple encoded field name between quotes",
            debug_print,
            jsonpath(testdata, '$..["book"][' " author.last.name " "][*][1]"),
            '["Jinn"]',
        )
        test_jdata(
            "jsonpath multiple encoded field name between []",
            debug_print,
            jsonpath(testdata, '$..["book"][ author.last.name ][*][1]'),
            '["Jinn"]',
        )
        test_jdata(
            "jsonpath escape . using \.",
            debug_print,
            jsonpath(testdata, "$.game\.arcade"),
            '{"title":"Mario"}',
        )
        test_jdata(
            "jsonpath escape . using []",
            debug_print,
            jsonpath(testdata, "$.[game.arcade]"),
            '{"title":"Mario"}',
        )
        test_jdata(
            "jsonpath scan struct array",
            debug_print,
            jsonpath(testdata, "$.book[*]..author[*]"),
            "null",
        )


if __name__ == "__main__":
    unittest.main()
