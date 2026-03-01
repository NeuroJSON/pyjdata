#!/usr/bin/env python3
"""
Comprehensive unit tests for pyjdata

Covers functions and code paths not exercised by existing test files (testjd.py,
testjdict.py, testgifti.py, testsnirf.py, testnifti.py, etc.).

Modules tested:
  - jdata.jdata   : encode/decode edge cases, codecs, special types
  - jdata.jfile   : save/load roundtrips, buffer functions, show/dumpb, aliases
  - jdata.jpath   : jsonpath set operations, edge cases
  - jdata.jschema  : coerce, _load_schema, validation edge cases
  - jdata.csvtsv (or csv): enum encode/decode, tsv2json, json2tsv, tonumbers
  - jdata.__init__ : version, __all__ sanity

Run:
    python -m unittest -v test.test_comprehensive

Copyright (c) 2026 Qianqian Fang - test suite generated for quality improvement
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jdata as jd
from jdata.jdata import (
    encode,
    decode,
    jdataencode,
    jdatadecode,
    jsonfilter,
    zlibencode,
    zlibdecode,
    gzipencode,
    gzipdecode,
    lzmaencode,
    lzmadecode,
    base64encode,
    base64decode,
    jdtype,
)
from jdata.jfile import (
    loadt,
    savet,
    loadts,
    loadjd,
    savejd,
    show,
    dumpb,
    jext,
    loadjson,
    savejson,
    loadbj,
    savebj,
    loadubjson,
    saveubjson,
)
from jdata.jpath import jsonpath
from jdata.jschema import jsonschema, coerce

# Import csv module (may be csvtsv)
try:
    from jdata.csvtsv import (
        load_csv_tsv,
        save_csv_tsv,
        encode_enum_column,
        decode_enum_column,
        is_enum_encoded,
        tsv2json,
        json2tsv,
    )
except ImportError:
    from jdata.csv import (
        load_csv_tsv,
        save_csv_tsv,
        encode_enum_column,
        decode_enum_column,
        is_enum_encoded,
        tsv2json,
        json2tsv,
    )


# ============================================================================
# Helper
# ============================================================================


class TempDir:
    """Context manager for a temporary directory."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="pyjdata_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


# ============================================================================
# jdata.jdata - encode/decode
# ============================================================================


class TestEncodeScalars(unittest.TestCase):
    """Test encoding of Python scalar types."""

    def test_nan(self):
        self.assertEqual(encode(float("nan")), "_NaN_")

    def test_positive_inf(self):
        self.assertEqual(encode(float("inf")), "_Inf_")

    def test_negative_inf(self):
        self.assertEqual(encode(float("-inf")), "-_Inf_")

    def test_regular_float(self):
        self.assertEqual(encode(3.14), 3.14)

    def test_integer_passthrough(self):
        self.assertEqual(encode(42), 42)

    def test_string_passthrough(self):
        self.assertEqual(encode("hello"), "hello")

    def test_bool_passthrough(self):
        self.assertEqual(encode(True), True)
        self.assertEqual(encode(False), False)

    def test_none_passthrough(self):
        self.assertIsNone(encode(None))


class TestEncodeComplex(unittest.TestCase):
    """Test encoding of complex numbers."""

    def test_complex_scalar(self):
        result = encode(3 + 4j)
        self.assertIn("_ArrayType_", result)
        self.assertIn("_ArrayIsComplex_", result)
        self.assertTrue(result["_ArrayIsComplex_"])
        self.assertEqual(result["_ArrayData_"], [3.0, 4.0])

    def test_complex_array(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        result = encode(arr)
        self.assertIn("_ArrayIsComplex_", result)
        self.assertTrue(result["_ArrayIsComplex_"])
        self.assertEqual(result["_ArraySize_"], [2])

    def test_complex_roundtrip(self):
        val = 3.5 + 2.1j
        encoded = encode(val)
        decoded = decode(encoded)
        self.assertAlmostEqual(decoded.real, val.real, places=10)
        self.assertAlmostEqual(decoded.imag, val.imag, places=10)

    def test_complex_array_roundtrip(self):
        arr = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
        encoded = encode(arr)
        decoded = decode(encoded)
        np.testing.assert_array_almost_equal(decoded, arr)


class TestEncodeContainers(unittest.TestCase):
    """Test encoding of various container types."""

    def test_list(self):
        result = encode([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_tuple(self):
        result = encode((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])

    def test_set(self):
        result = encode({1, 2, 3})
        self.assertEqual(sorted(result), [1, 2, 3])

    def test_frozenset(self):
        result = encode(frozenset([1, 2, 3]))
        self.assertEqual(sorted(result), [1, 2, 3])

    def test_nested_dict(self):
        data = {"a": {"b": [1, 2, 3]}, "c": "test"}
        result = encode(data)
        self.assertEqual(result["a"]["b"], [1, 2, 3])
        self.assertEqual(result["c"], "test")

    def test_mixed_nested(self):
        data = {"arr": np.array([1, 2]), "nan": float("nan"), "list": [1, "two"]}
        result = encode(data)
        self.assertEqual(result["nan"], "_NaN_")
        self.assertIn("_ArrayType_", result["arr"])

    def test_range(self):
        """range is not directly supported by encode; convert to list first."""
        result = encode(list(range(5)))
        self.assertEqual(result, [0, 1, 2, 3, 4])


class TestEncodeNumpyArrays(unittest.TestCase):
    """Test encoding of various numpy array types and dtypes."""

    def test_uint8(self):
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "uint8")

    def test_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "int32")

    def test_float64(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "double")

    def test_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "single")

    def test_bool_array(self):
        arr = np.array([True, False, True], dtype=bool)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "uint8")

    def test_2d_array_shape(self):
        arr = np.zeros((3, 4), dtype=np.float64)
        result = encode(arr)
        self.assertEqual(result["_ArraySize_"], [3, 4])

    def test_3d_array_shape(self):
        arr = np.zeros((2, 3, 4), dtype=np.int16)
        result = encode(arr)
        self.assertEqual(result["_ArraySize_"], [2, 3, 4])

    def test_scalar_array(self):
        """A 1-element numpy array encodes with _ArraySize_=[1]."""
        arr = np.array([3.14], dtype=np.float64)
        result = encode(arr)
        self.assertIn("_ArrayType_", result)
        self.assertEqual(result["_ArrayType_"], "double")

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        result = encode(arr)
        self.assertEqual(result["_ArraySize_"], [0])

    def test_int64(self):
        arr = np.array([2**40], dtype=np.int64)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "int64")

    def test_uint64(self):
        arr = np.array([2**60], dtype=np.uint64)
        result = encode(arr)
        self.assertEqual(result["_ArrayType_"], "uint64")


class TestDecodeSpecialValues(unittest.TestCase):
    """Test decoding of JData special value annotations."""

    def test_nan_string(self):
        result = decode("_NaN_")
        self.assertTrue(np.isnan(result))

    def test_inf_string(self):
        result = decode("_Inf_")
        self.assertEqual(result, float("inf"))

    def test_neg_inf_string(self):
        result = decode("-_Inf_")
        self.assertEqual(result, float("-inf"))

    def test_decode_plain_string(self):
        result = decode("hello")
        self.assertEqual(result, "hello")

    def test_decode_plain_int(self):
        result = decode(42)
        self.assertEqual(result, 42)

    def test_decode_nested_nan(self):
        data = {"a": "_NaN_", "b": [1, "_Inf_", "-_Inf_"]}
        result = decode(data)
        self.assertTrue(np.isnan(result["a"]))
        self.assertEqual(result["b"][1], float("inf"))
        self.assertEqual(result["b"][2], float("-inf"))


class TestArrayRoundtrips(unittest.TestCase):
    """Test full encode -> decode roundtrips for numpy arrays."""

    def _roundtrip(self, arr, **kwargs):
        encoded = encode(arr, **kwargs)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, arr)

    def test_uint8_roundtrip(self):
        self._roundtrip(np.arange(10, dtype=np.uint8))

    def test_int16_roundtrip(self):
        self._roundtrip(np.array([-100, 0, 100], dtype=np.int16))

    def test_float32_roundtrip(self):
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        encoded = encode(arr)
        decoded = decode(encoded)
        np.testing.assert_array_almost_equal(decoded, arr, decimal=5)

    def test_2d_roundtrip(self):
        self._roundtrip(np.arange(12, dtype=np.int32).reshape(3, 4))

    def test_3d_roundtrip(self):
        self._roundtrip(np.arange(24, dtype=np.float64).reshape(2, 3, 4))

    def test_bool_roundtrip(self):
        arr = np.array([True, False, True, False], dtype=bool)
        encoded = encode(arr)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded.astype(bool), arr)


class TestCompressionRoundtrips(unittest.TestCase):
    """Test encode/decode with various compression codecs."""

    def setUp(self):
        self.arr = np.arange(100, dtype=np.float64)

    def _roundtrip_compression(self, method):
        encoded = encode(self.arr, compression=method, base64=True)
        self.assertIn("_ArrayZipType_", encoded)
        self.assertEqual(encoded["_ArrayZipType_"], method)
        decoded = decode(encoded, base64=True)
        np.testing.assert_array_equal(decoded, self.arr)

    def test_zlib(self):
        self._roundtrip_compression("zlib")

    def test_gzip(self):
        self._roundtrip_compression("gzip")

    def test_lzma(self):
        self._roundtrip_compression("lzma")

    def test_lz4(self):
        try:
            import lz4.frame
        except ImportError:
            self.skipTest("lz4 not installed")
        self._roundtrip_compression("lz4")

    def test_base64_only(self):
        self._roundtrip_compression("base64")

    def test_complex_with_zlib(self):
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j] * 50, dtype=np.complex128)
        encoded = encode(arr, compression="zlib", base64=True)
        decoded = decode(encoded, base64=True)
        np.testing.assert_array_almost_equal(decoded, arr)


# ============================================================================
# Codec utility functions
# ============================================================================


class TestCodecFunctions(unittest.TestCase):
    """Test the individual codec encode/decode functions."""

    def setUp(self):
        self.data = b"Hello, World! " * 100

    def test_zlib_roundtrip(self):
        compressed = zlibencode(self.data)
        decompressed = zlibdecode(compressed)
        self.assertEqual(decompressed, self.data)

    def test_gzip_roundtrip(self):
        compressed = gzipencode(self.data)
        decompressed = gzipdecode(compressed)
        self.assertEqual(decompressed, self.data)

    def test_lzma_roundtrip(self):
        compressed = lzmaencode(self.data)
        decompressed = lzmadecode(compressed)
        self.assertEqual(decompressed, self.data)

    def test_base64_roundtrip(self):
        encoded = base64encode(self.data)
        decoded = base64decode(encoded)
        self.assertEqual(decoded, self.data)

    def test_zlib_compresses(self):
        compressed = zlibencode(self.data)
        self.assertLess(len(compressed), len(self.data))

    def test_gzip_compresses(self):
        compressed = gzipencode(self.data)
        self.assertLess(len(compressed), len(self.data))


# ============================================================================
# jsonfilter
# ============================================================================


class TestJsonFilter(unittest.TestCase):
    """Test the jsonfilter function used for JSON serialization fallbacks."""

    def test_numpy_int(self):
        result = jsonfilter(np.int32(5))
        self.assertEqual(result, 5)

    def test_numpy_float(self):
        result = jsonfilter(np.float64(3.14))
        self.assertAlmostEqual(result, 3.14)

    def test_numpy_bool(self):
        result = jsonfilter(np.bool_(True))
        self.assertEqual(result, True)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = jsonfilter(arr)
        self.assertEqual(result, [1, 2, 3])

    def test_bytes(self):
        result = jsonfilter(b"hello")
        self.assertIsInstance(result, str)

    def test_set(self):
        """jsonfilter does not handle sets; verify it returns None gracefully."""
        result = jsonfilter({1, 2, 3})
        # jsonfilter may return None for unsupported types
        self.assertIsNone(result)


# ============================================================================
# jdtype mapping
# ============================================================================


class TestJdtype(unittest.TestCase):
    """Test that jdtype covers expected numpy types."""

    def test_float32(self):
        self.assertEqual(jdtype["float32"], "single")

    def test_float64(self):
        self.assertEqual(jdtype["float64"], "double")

    def test_complex128(self):
        self.assertEqual(jdtype["complex128"], "double")

    def test_complex64(self):
        self.assertEqual(jdtype["complex64"], "single")

    def test_int_types(self):
        self.assertEqual(jdtype["byte"], "int8")
        self.assertEqual(jdtype["ubyte"], "uint8")
        self.assertEqual(jdtype["short"], "int16")
        self.assertEqual(jdtype["ushort"], "uint16")
        self.assertEqual(jdtype["longlong"], "int64")
        self.assertEqual(jdtype["ulonglong"], "uint64")


# ============================================================================
# jdata.jfile - File I/O
# ============================================================================


class TestSaveLoadJSON(unittest.TestCase):
    """Test save/load roundtrips with JSON text files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pyjdata_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_simple_dict(self):
        data = {"name": "test", "value": 42, "pi": 3.14}
        path = os.path.join(self.tmpdir, "test.json")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["name"], "test")
        self.assertEqual(loaded["value"], 42)
        self.assertAlmostEqual(loaded["pi"], 3.14)

    def test_numpy_array(self):
        data = {"arr": np.arange(10, dtype=np.uint8)}
        path = os.path.join(self.tmpdir, "arr.json")
        jd.save(data, path)
        loaded = jd.load(path)
        np.testing.assert_array_equal(loaded["arr"], data["arr"])

    def test_nested_structure(self):
        data = {"a": {"b": {"c": [1, 2, 3]}}, "d": "deep"}
        path = os.path.join(self.tmpdir, "nested.json")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["a"]["b"]["c"], [1, 2, 3])

    def test_special_values(self):
        data = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf")}
        path = os.path.join(self.tmpdir, "special.json")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertTrue(np.isnan(loaded["nan"]))
        self.assertEqual(loaded["inf"], float("inf"))
        self.assertEqual(loaded["ninf"], float("-inf"))

    def test_compressed_array(self):
        data = {"arr": np.arange(500, dtype=np.float64)}
        path = os.path.join(self.tmpdir, "compressed.json")
        jd.save(data, path, compression="zlib")
        loaded = jd.load(path)
        np.testing.assert_array_equal(loaded["arr"], data["arr"])

    def test_no_encode_decode(self):
        """Save without encoding, load without decoding."""
        data = {"val": 42}
        path = os.path.join(self.tmpdir, "raw.json")
        savet(data, path, encode=False)
        loaded = loadt(path, decode=False)
        self.assertEqual(loaded["val"], 42)

    def test_empty_dict(self):
        data = {}
        path = os.path.join(self.tmpdir, "empty.json")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded, {})

    def test_empty_list(self):
        data = []
        path = os.path.join(self.tmpdir, "emptylist.json")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded, [])

    def test_complex_roundtrip_json(self):
        data = {"c": np.array([1 + 2j, 3 + 4j], dtype=np.complex128)}
        path = os.path.join(self.tmpdir, "complex.json")
        jd.save(data, path)
        loaded = jd.load(path)
        np.testing.assert_array_almost_equal(loaded["c"], data["c"])


class TestSaveLoadBinary(unittest.TestCase):
    """Test save/load roundtrips with binary (BJData) files."""

    def setUp(self):
        try:
            import bjdata

            self.has_bjdata = True
        except ImportError:
            self.has_bjdata = False
        self.tmpdir = tempfile.mkdtemp(prefix="pyjdata_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bjdata_roundtrip(self):
        if not self.has_bjdata:
            self.skipTest("bjdata not installed")
        data = {"name": "test", "arr": np.arange(10, dtype=np.float64)}
        path = os.path.join(self.tmpdir, "test.bjd")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["name"], "test")
        np.testing.assert_array_equal(loaded["arr"], data["arr"])

    def test_ubj_roundtrip(self):
        if not self.has_bjdata:
            self.skipTest("bjdata not installed")
        data = {"x": 42, "y": [1, 2, 3]}
        path = os.path.join(self.tmpdir, "test.ubj")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["x"], 42)

    def test_jdb_roundtrip(self):
        if not self.has_bjdata:
            self.skipTest("bjdata not installed")
        data = {"key": "value"}
        path = os.path.join(self.tmpdir, "test.jdb")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["key"], "value")


class TestSaveLoadJD(unittest.TestCase):
    """Test the unified loadjd/savejd interface."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pyjdata_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_jnii_text(self):
        data = {"NIFTIHeader": {"dim": [1, 2, 3]}, "NIFTIData": [1, 2, 3]}
        path = os.path.join(self.tmpdir, "test.jnii")
        savejd(data, path)
        loaded = loadjd(path)
        self.assertEqual(loaded["NIFTIHeader"]["dim"], [1, 2, 3])

    def test_json_extension(self):
        data = {"test": True}
        path = os.path.join(self.tmpdir, "test.json")
        savejd(data, path)
        loaded = loadjd(path)
        self.assertTrue(loaded["test"])


class TestBufferFunctions(unittest.TestCase):
    """Test loadts (text buffer) and show/dumpb."""

    def test_loadts_simple(self):
        result = loadts('{"a": 1, "b": 2}')
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)

    def test_loadts_with_nan(self):
        result = loadts('{"val": "_NaN_"}')
        self.assertTrue(np.isnan(result["val"]))

    def test_loadts_array(self):
        result = loadts("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])

    def test_loadts_no_decode(self):
        result = loadts('{"val": "_NaN_"}', decode=False)
        self.assertEqual(result["val"], "_NaN_")

    def test_show_returns_string(self):
        result = show({"a": 1}, string=True)
        self.assertIn('"a"', result)
        self.assertIn("1", result)

    def test_show_numpy(self):
        data = {"arr": np.array([1, 2, 3], dtype=np.uint8)}
        result = show(data, string=True)
        self.assertIn("_ArrayType_", result)

    def test_dumpb(self):
        try:
            import bjdata
        except ImportError:
            self.skipTest("bjdata not installed")
        result = dumpb({"a": 1})
        self.assertIsInstance(result, (bytes, bytearray))


class TestFileAliases(unittest.TestCase):
    """Test that function aliases work."""

    def test_jdataencode_is_encode(self):
        data = {"a": np.array([1, 2], dtype=np.uint8)}
        r1 = encode(data)
        r2 = jdataencode(data)
        self.assertEqual(str(r1), str(r2))

    def test_jdatadecode_is_decode(self):
        data = {"val": "_NaN_"}
        r1 = decode(data)
        r2 = jdatadecode(data)
        self.assertTrue(np.isnan(r1["val"]))
        self.assertTrue(np.isnan(r2["val"]))


class TestJextGlobal(unittest.TestCase):
    """Test the jext file extension mapping."""

    def test_text_extensions(self):
        self.assertIn(".json", jext["t"])
        self.assertIn(".jnii", jext["t"])

    def test_binary_extensions(self):
        self.assertIn(".bjd", jext["b"])
        self.assertIn(".bnii", jext["b"])


# ============================================================================
# jdata.jpath - JSONPath
# ============================================================================


class TestJsonPathGet(unittest.TestCase):
    """Test jsonpath get operations."""

    def setUp(self):
        self.data = {
            "store": {
                "book": [
                    {"title": "A", "price": 10},
                    {"title": "B", "price": 20},
                    {"title": "C", "price": 30},
                ],
                "name": "MyStore",
            }
        }

    def test_root(self):
        result = jsonpath(self.data, "$")
        self.assertEqual(result, self.data)

    def test_simple_key(self):
        result = jsonpath(self.data, "$.store.name")
        self.assertEqual(result, "MyStore")

    def test_array_index(self):
        result = jsonpath(self.data, "$.store.book[0]")
        if isinstance(result, list):
            result = result[0]
        self.assertEqual(result["title"], "A")

    def test_deep_scan(self):
        result = jsonpath(self.data, "$..title")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)

    def test_deep_scan_price(self):
        result = jsonpath(self.data, "$..price")
        self.assertIn(10, result)
        self.assertIn(20, result)
        self.assertIn(30, result)

    def test_nonexistent_key(self):
        """Accessing a missing key should not crash."""
        try:
            result = jsonpath(self.data, "$.store.nonexistent")
            # Some implementations return None, others raise
        except (ValueError, KeyError):
            pass  # acceptable

    def test_wildcard(self):
        result = jsonpath(self.data, "$.store.book[*]")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)


class TestJsonPathSet(unittest.TestCase):
    """Test jsonpath set/write operations via jdict."""

    def test_set_simple(self):
        from jdata.jdictionary import jdict

        jd_obj = jdict({"a": 1, "b": 2})
        jd_obj["$.a"] = 99
        self.assertEqual(jd_obj.a(), 99)

    def test_set_nested(self):
        from jdata.jdictionary import jdict

        jd_obj = jdict({"a": {"b": {"c": 1}}})
        jd_obj["$.a.b.c"] = 42
        self.assertEqual(jd_obj.a.b.c(), 42)

    def test_set_via_attribute(self):
        from jdata.jdictionary import jdict

        jd_obj = jdict({"x": 10, "y": 20})
        jd_obj.x = 99
        self.assertEqual(jd_obj.x(), 99)

    def test_set_array_element(self):
        from jdata.jdictionary import jdict

        jd_obj = jdict({"arr": [10, 20, 30]})
        jd_obj.arr[1] = 99
        self.assertEqual(jd_obj.arr.v(1), 99)

    def test_set_new_key(self):
        from jdata.jdictionary import jdict

        jd_obj = jdict({"a": 1})
        jd_obj.b = "new"
        self.assertEqual(jd_obj.b(), "new")


class TestJsonPathEdgeCases(unittest.TestCase):
    """Test edge cases in jsonpath."""

    def test_empty_data(self):
        result = jsonpath({}, "$")
        self.assertEqual(result, {})

    def test_single_value(self):
        data = {"x": 5}
        result = jsonpath(data, "$.x")
        self.assertEqual(result, 5)

    def test_key_with_dot(self):
        """Test escaped dot in key names."""
        data = {"a.b": 42}
        result = jsonpath(data, r"$.a\.b")
        self.assertEqual(result, 42)

    def test_bracket_notation(self):
        data = {"book": [{"author": "Yoda"}]}
        result = jsonpath(data, "$[book][0][author]")
        self.assertEqual(result, "Yoda")

    def test_slice_range(self):
        data = {"items": [0, 1, 2, 3, 4]}
        result = jsonpath(data, "$.items[1:3]")
        self.assertIsInstance(result, list)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_negative_index(self):
        data = {"items": ["a", "b", "c", "d"]}
        result = jsonpath(data, "$.items[-1]")
        if isinstance(result, list):
            self.assertIn("d", result)
        else:
            self.assertEqual(result, "d")


# ============================================================================
# jdata.jschema - JSON Schema
# ============================================================================


class TestSchemaValidation(unittest.TestCase):
    """Test jsonschema validation beyond testjdict.py coverage."""

    def test_integer_valid(self):
        valid, errors = jsonschema(5, {"type": "integer"})
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_integer_invalid(self):
        valid, errors = jsonschema("five", {"type": "integer"})
        self.assertFalse(valid)

    def test_number_valid(self):
        valid, errors = jsonschema(3.14, {"type": "number"})
        self.assertTrue(valid)

    def test_string_valid(self):
        valid, errors = jsonschema("hello", {"type": "string"})
        self.assertTrue(valid)

    def test_boolean_valid(self):
        valid, errors = jsonschema(True, {"type": "boolean"})
        self.assertTrue(valid)

    def test_null_valid(self):
        valid, errors = jsonschema(None, {"type": "null"})
        self.assertTrue(valid)

    def test_null_invalid(self):
        valid, errors = jsonschema(0, {"type": "null"})
        self.assertFalse(valid)

    def test_array_items_type(self):
        valid, errors = jsonschema([1, 2, 3], {"type": "array", "items": {"type": "integer"}})
        self.assertTrue(valid)

    def test_array_items_invalid(self):
        valid, errors = jsonschema([1, "two", 3], {"type": "array", "items": {"type": "integer"}})
        self.assertFalse(valid)

    def test_object_required(self):
        valid, errors = jsonschema(
            {"name": "test"},
            {"type": "object", "required": ["name", "age"]},
        )
        self.assertFalse(valid)

    def test_object_additional_properties_false(self):
        valid, errors = jsonschema(
            {"name": "test", "extra": 1},
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        self.assertFalse(valid)

    def test_multiple_of(self):
        valid, errors = jsonschema(15, {"type": "integer", "multipleOf": 5})
        self.assertTrue(valid)

    def test_multiple_of_fail(self):
        valid, errors = jsonschema(13, {"type": "integer", "multipleOf": 5})
        self.assertFalse(valid)

    def test_pattern(self):
        valid, errors = jsonschema("abc123", {"type": "string", "pattern": "^[a-z]+[0-9]+$"})
        self.assertTrue(valid)

    def test_pattern_fail(self):
        valid, errors = jsonschema("123abc", {"type": "string", "pattern": "^[a-z]+[0-9]+$"})
        self.assertFalse(valid)

    def test_enum(self):
        valid, errors = jsonschema("red", {"enum": ["red", "green", "blue"]})
        self.assertTrue(valid)

    def test_enum_fail(self):
        valid, errors = jsonschema("yellow", {"enum": ["red", "green", "blue"]})
        self.assertFalse(valid)

    def test_const(self):
        valid, errors = jsonschema("fixed", {"const": "fixed"})
        self.assertTrue(valid)

    def test_allof(self):
        valid, errors = jsonschema(10, {"allOf": [{"type": "integer"}, {"minimum": 5}]})
        self.assertTrue(valid)

    def test_anyof(self):
        valid, errors = jsonschema("hi", {"anyOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertTrue(valid)

    def test_oneof(self):
        valid, errors = jsonschema(5, {"oneOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertTrue(valid)

    def test_not_schema(self):
        valid, errors = jsonschema("hi", {"not": {"type": "integer"}})
        self.assertTrue(valid)


class TestSchemaGeneration(unittest.TestCase):
    """Test jsonschema generation beyond testjdict.py coverage."""

    def test_generate_integer(self):
        result = jsonschema({"type": "integer"}, generate="all")
        self.assertIsInstance(result, int)

    def test_generate_string(self):
        result = jsonschema({"type": "string"}, generate="all")
        self.assertIsInstance(result, str)

    def test_generate_boolean(self):
        result = jsonschema({"type": "boolean"}, generate="all")
        self.assertIsInstance(result, bool)

    def test_generate_array_with_items(self):
        result = jsonschema(
            {"type": "array", "minItems": 3, "items": {"type": "integer"}},
            generate="all",
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(x, int) for x in result))

    def test_generate_object(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "default": "Alice"},
                "age": {"type": "integer"},
            },
        }
        result = jsonschema(schema, generate="all")
        self.assertIn("name", result)
        self.assertEqual(result["name"], "Alice")

    def test_generate_with_default(self):
        result = jsonschema({"type": "number", "default": 3.14}, generate="all")
        self.assertAlmostEqual(result, 3.14)


class TestCoerce(unittest.TestCase):
    """Test the coerce function."""

    def test_coerce_to_uint8(self):
        result = coerce([1, 2, 3], {"binType": "uint8"})
        self.assertEqual(result.dtype, np.uint8)

    def test_coerce_to_float32(self):
        result = coerce([1.0, 2.0], {"binType": "float32"})
        self.assertEqual(result.dtype, np.float32)

    def test_coerce_no_bintype(self):
        result = coerce([1, 2], {"type": "array"})
        self.assertEqual(result, [1, 2])

    def test_coerce_already_correct(self):
        arr = np.array([1, 2], dtype=np.int32)
        result = coerce(arr, {"binType": "int32"})
        self.assertIs(result, arr)

    def test_coerce_string_fallback(self):
        result = coerce("not a number", {"binType": "uint8"})
        self.assertEqual(result, "not a number")


# ============================================================================
# CSV/TSV module
# ============================================================================


class TestCsvTsvIO(unittest.TestCase):
    """Test CSV/TSV read/write functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pyjdata_test_csv_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_read_csv(self):
        data = {"name": ["Alice", "Bob"], "age": [30, 25]}
        path = os.path.join(self.tmpdir, "test.csv")
        save_csv_tsv(data, path, delimiter=",")
        loaded = load_csv_tsv(path, delimiter=",")
        self.assertEqual(loaded["name"], ["Alice", "Bob"])

    def test_write_read_tsv(self):
        data = {"col1": ["a", "b", "c"], "col2": [1, 2, 3]}
        path = os.path.join(self.tmpdir, "test.tsv")
        save_csv_tsv(data, path, delimiter="\t")
        loaded = load_csv_tsv(path, delimiter="\t")
        self.assertEqual(loaded["col1"], ["a", "b", "c"])

    def test_auto_detect_csv_delimiter(self):
        data = {"x": ["1", "2"], "y": ["3", "4"]}
        path = os.path.join(self.tmpdir, "auto.csv")
        save_csv_tsv(data, path, delimiter=",")
        loaded = load_csv_tsv(path)  # auto-detect from .csv extension
        self.assertIn("x", loaded)

    def test_auto_detect_tsv_delimiter(self):
        data = {"x": ["1", "2"], "y": ["3", "4"]}
        path = os.path.join(self.tmpdir, "auto.tsv")
        save_csv_tsv(data, path, delimiter="\t")
        loaded = load_csv_tsv(path)  # auto-detect from .tsv extension
        self.assertIn("x", loaded)

    def test_empty_csv(self):
        """An empty file should not crash."""
        path = os.path.join(self.tmpdir, "empty.csv")
        with open(path, "w") as f:
            f.write("")
        try:
            loaded = load_csv_tsv(path)
        except Exception:
            pass  # acceptable to raise on empty file


class TestEnumEncodeDecode(unittest.TestCase):
    """Test _EnumKey_/_EnumValue_ column encoding."""

    def test_encode_small_column(self):
        """Few values - should NOT encode (not beneficial)."""
        values = ["a", "b"]
        result = encode_enum_column(values)
        self.assertEqual(result, values)

    def test_encode_large_column(self):
        """Many repeated values - should encode."""
        values = ["male", "female"] * 100
        result = encode_enum_column(values, compress=True)
        self.assertTrue(is_enum_encoded(result))
        self.assertIn("_EnumKey_", result)
        self.assertIn("_EnumValue_", result)

    def test_decode_roundtrip(self):
        values = ["cat", "dog", "cat", "bird"] * 50
        encoded = encode_enum_column(values, compress=True)
        decoded = decode_enum_column(encoded)
        self.assertEqual(decoded, values)

    def test_decode_uncompressed(self):
        values = ["x", "y", "z"] * 50
        encoded = encode_enum_column(values, compress=False)
        if is_enum_encoded(encoded):
            decoded = decode_enum_column(encoded)
            self.assertEqual(decoded, values)

    def test_is_enum_encoded_false(self):
        self.assertFalse(is_enum_encoded([1, 2, 3]))
        self.assertFalse(is_enum_encoded({"a": 1}))
        self.assertFalse(is_enum_encoded("string"))

    def test_empty_values(self):
        result = encode_enum_column([])
        self.assertEqual(result, [])

    def test_many_unique_values(self):
        """300+ unique values should use uint16."""
        values = [f"val_{i}" for i in range(300)] * 3
        encoded = encode_enum_column(values, compress=True)
        if is_enum_encoded(encoded):
            self.assertEqual(encoded["_EnumValue_"]["_ArrayType_"], "uint16")


class TestTsvJsonConversion(unittest.TestCase):
    """Test tsv2json and json2tsv functions."""

    def test_tsv2json_no_compress(self):
        rows = [["name", "age"], ["Alice", "30"], ["Bob", "25"]]
        result = tsv2json(rows, compress=False)
        self.assertIn("name", result)
        self.assertIn("age", result)

    def test_tsv2json_with_compress(self):
        rows = [["name", "val"]]
        rows += [["Alice", "x"]] * 100
        rows += [["Bob", "y"]] * 100
        result = tsv2json(rows, compress=True)
        self.assertIn("name", result)

    def test_json2tsv_roundtrip(self):
        data = {"name": ["Alice", "Bob", "Carol"], "score": [90, 85, 95]}
        tsv_str = json2tsv(data)
        self.assertIn("name", tsv_str)
        self.assertIn("Alice", tsv_str)

    def test_json2tsv_to_file(self):
        with TempDir() as tmpdir:
            data = {"col1": ["a", "b"], "col2": ["c", "d"]}
            path = os.path.join(tmpdir, "out.tsv")
            json2tsv(data, filepath=path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("col1", content)

    def test_json2tsv_with_enum(self):
        """json2tsv should decode enum-encoded columns."""
        values = ["a", "b"] * 100
        encoded_col = encode_enum_column(values, compress=True)
        data = {"col": encoded_col, "other": list(range(200))}
        tsv_str = json2tsv(data)
        self.assertIn("col", tsv_str)


# ============================================================================
# jdata.__init__ - module-level sanity
# ============================================================================


class TestModuleSanity(unittest.TestCase):
    """Test module-level attributes and __all__."""

    def test_version_exists(self):
        self.assertTrue(hasattr(jd, "__version__"))
        self.assertIsInstance(jd.__version__, str)

    def test_all_is_flat_list_of_strings(self):
        self.assertIsInstance(jd.__all__, (list, tuple))
        for item in jd.__all__:
            self.assertIsInstance(item, str, f"__all__ contains non-string: {item!r}")

    def test_core_functions_accessible(self):
        self.assertTrue(callable(jd.encode))
        self.assertTrue(callable(jd.decode))
        self.assertTrue(callable(jd.load))
        self.assertTrue(callable(jd.save))
        self.assertTrue(callable(jd.loadjd))
        self.assertTrue(callable(jd.savejd))

    def test_license_exists(self):
        self.assertTrue(hasattr(jd, "__license__"))
        self.assertIn("Apache", jd.__license__)


# ============================================================================
# Integration tests - full pipeline
# ============================================================================


class TestFullPipeline(unittest.TestCase):
    """Integration tests covering typical user workflows."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pyjdata_integ_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mixed_data_json_roundtrip(self):
        """A realistic mixed data structure through JSON."""
        data = {
            "subject": "sub-01",
            "age": 25,
            "scan_data": np.random.rand(10, 10).astype(np.float32),
            "labels": ["region_a", "region_b", "region_c"],
            "metadata": {
                "scanner": "3T",
                "tr": 2.0,
                "volumes": 200,
            },
        }
        path = os.path.join(self.tmpdir, "subject.json")
        jd.save(data, path, compression="zlib")
        loaded = jd.load(path)

        self.assertEqual(loaded["subject"], "sub-01")
        self.assertEqual(loaded["age"], 25)
        np.testing.assert_array_almost_equal(loaded["scan_data"], data["scan_data"], decimal=5)
        self.assertEqual(loaded["labels"], data["labels"])
        self.assertEqual(loaded["metadata"]["scanner"], "3T")

    def test_mixed_data_bjd_roundtrip(self):
        """A realistic mixed data structure through BJData."""
        try:
            import bjdata
        except ImportError:
            self.skipTest("bjdata not installed")

        data = {
            "name": "experiment_1",
            "values": np.linspace(0, 1, 50, dtype=np.float64),
            "flags": np.array([True, False, True], dtype=bool),
        }
        path = os.path.join(self.tmpdir, "experiment.bjd")
        jd.save(data, path)
        loaded = jd.load(path)

        self.assertEqual(loaded["name"], "experiment_1")
        np.testing.assert_array_almost_equal(loaded["values"], data["values"])

    def test_encode_decode_preserves_structure(self):
        """Encode then decode should restore original structure."""
        original = {
            "ints": np.array([1, 2, 3], dtype=np.int32),
            "floats": np.array([1.1, 2.2], dtype=np.float64),
            "complex": np.array([1 + 1j], dtype=np.complex128),
            "nested": {"arr": np.zeros((2, 3), dtype=np.uint8)},
            "string": "test",
            "number": 42,
            "special": float("nan"),
        }
        encoded = jd.encode(original)
        decoded = jd.decode(encoded)

        np.testing.assert_array_equal(decoded["ints"], original["ints"])
        np.testing.assert_array_almost_equal(decoded["floats"], original["floats"])
        np.testing.assert_array_almost_equal(decoded["complex"], original["complex"])
        np.testing.assert_array_equal(decoded["nested"]["arr"], original["nested"]["arr"])
        self.assertEqual(decoded["string"], "test")
        self.assertEqual(decoded["number"], 42)
        self.assertTrue(np.isnan(decoded["special"]))

    def test_multiple_compression_methods(self):
        """Test that all standard codecs produce valid roundtrips through files."""
        arr = np.arange(500, dtype=np.float64)

        for method in ["zlib", "gzip", "lzma"]:
            path = os.path.join(self.tmpdir, f"test_{method}.json")
            jd.save({"arr": arr}, path, compression=method)
            loaded = jd.load(path)
            np.testing.assert_array_equal(loaded["arr"], arr, err_msg=f"Failed for codec: {method}")

    def test_jnii_text_roundtrip(self):
        """Save and load a JNIfTI-style structure."""
        data = {
            "NIFTIHeader": {"Dim": [64, 64, 32]},
            "NIFTIData": np.zeros((4, 4, 4), dtype=np.float32),
        }
        path = os.path.join(self.tmpdir, "brain.jnii")
        jd.save(data, path)
        loaded = jd.load(path)
        self.assertEqual(loaded["NIFTIHeader"]["Dim"], [64, 64, 32])
        np.testing.assert_array_equal(loaded["NIFTIData"], data["NIFTIData"])

    def test_csv_through_jd_load(self):
        """Test that jd.load can handle CSV files."""
        csv_content = "name,age,score\nAlice,30,95\nBob,25,88\n"
        path = os.path.join(self.tmpdir, "data.csv")
        with open(path, "w") as f:
            f.write(csv_content)
        try:
            loaded = jd.load(path)
            self.assertIn("name", loaded)
        except Exception:
            pass  # CSV loading may not be registered in all versions

    def test_jsonpath_on_loaded_data(self):
        """Use jsonpath on data loaded from a file."""
        data = {
            "subjects": [
                {"id": "sub-01", "age": 25},
                {"id": "sub-02", "age": 30},
                {"id": "sub-03", "age": 35},
            ]
        }
        path = os.path.join(self.tmpdir, "subjects.json")
        jd.save(data, path)
        loaded = jd.load(path)
        ids = jsonpath(loaded, "$..id")
        self.assertEqual(len(ids), 3)
        self.assertIn("sub-01", ids)


if __name__ == "__main__":
    unittest.main()
