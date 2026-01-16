"""
Unit tests for jdict and jsonschema modules
Translated from MATLAB jsonlab test suite

Author: Qianqian Fang (q.fang at neu.edu)
"""

import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.jdict import jdict
from jdata.jschema import jsonschema


class TestJdictBasic(unittest.TestCase):
    def setUp(self):
        self.testdata = {"key1": {"subkey1": 1, "subkey2": [1, 2, 3]}, "subkey2": "str"}
        self.testdata["key1"]["subkey3"] = [8, "test", {"subsubkey1": 0}]

    def test_basic_access(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey1(), 1)

    def test_subkey_access(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey3(), [8, "test", {"subsubkey1": 0}])

    def test_v_method_single(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey3.v(0), 8)

    def test_v_method_nested(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey3.v(2)["subsubkey1"], 0)

    def test_v_method_string(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey3.v(1), "test")

    def test_jsonpath_basic(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd["$.key1.subkey1"](), 1)

    def test_jsonpath_array(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd["$.key1.subkey2"](), [1, 2, 3])

    def test_jsonpath_array_index(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd["$.key1.subkey3[2].subsubkey1"](), 0)

    def test_deep_scan(self):
        jd = jdict(self.testdata)
        result = jd["$..subkey2"]()
        self.assertEqual(len(result), 4)

    def test_assignment_subkey(self):
        jd = jdict(self.testdata)
        jd.key1.subkey2 = [2, 10, 11]
        self.assertEqual(jd.key1.subkey2(), [2, 10, 11])

    def test_assignment_nested(self):
        jd = jdict(self.testdata)
        jd.key1.subkey3()[2]["subsubkey1"] = 1
        self.assertEqual(jd.key1.subkey3.v(2)["subsubkey1"], 1)

    def test_keys_struct(self):
        jd = jdict(self.testdata)
        self.assertIn("subkey1", jd.key1.keys())
        self.assertIn("subkey2", jd.key1.keys())
        self.assertIn("subkey3", jd.key1.keys())

    def test_len(self):
        jd = jdict(self.testdata)
        self.assertEqual(jd.key1.subkey2.len(), 3)

    def test_iskey_exists(self):
        jd = jdict(self.testdata)
        self.assertTrue(jd.key1.iskey("subkey3"))

    def test_iskey_not_exists(self):
        jd = jdict(self.testdata)
        self.assertFalse(jd.key1.iskey("subkey4"))


class TestJdictAdvanced(unittest.TestCase):
    def test_empty_constructor(self):
        jd = jdict()
        self.assertIsNone(jd.v())

    def test_add_to_empty(self):
        jd = jdict()
        jd.newkey = "newvalue"
        self.assertEqual(jd.newkey(), "newvalue")

    def test_copy_constructor(self):
        jd1 = jdict({"a": 1, "b": 2})
        jd1.setattr("$.a", "myattr", "attrvalue")
        jd2 = jdict(jd1)
        self.assertEqual(jd2.a(), 1)
        self.assertEqual(jd2.getattr("$.a", "myattr"), "attrvalue")

    def test_tojson(self):
        jd = jdict({"name": "test", "value": 123})
        js = jd.tojson()
        self.assertIn("name", js)
        self.assertIn("test", js)

    def test_keys_array(self):
        jd = jdict([1, 2, 3, 4, 5])
        self.assertEqual(jd.keys(), [0, 1, 2, 3, 4])

    def test_size_array(self):
        jd = jdict([1, 2, 3, 4, 5])
        self.assertEqual(jd.size(), (5,))

    def test_v_returns_data(self):
        jd = jdict([10, 20, 30, 40])
        self.assertEqual(jd.v(), [10, 20, 30, 40])

    def test_v_single_index(self):
        jd = jdict([10, 20, 30, 40])
        self.assertEqual(jd.v(1), 20)

    def test_v_slice(self):
        jd = jdict([10, 20, 30, 40])
        self.assertEqual(jd.v(slice(1, 3)), [20, 30])


class TestJdictAttributes(unittest.TestCase):
    def test_root_level_dims(self):
        jd = jdict([1, 2, 3])
        jd.setattr("dims", ["time", "channels", "trials"])
        self.assertEqual(jd.getattr("dims"), ["time", "channels", "trials"])

    def test_root_level_units(self):
        jd = jdict([1, 2, 3])
        jd.setattr("units", "uV")
        self.assertEqual(jd.getattr("units"), "uV")

    def test_multiple_attrs(self):
        jd = jdict([1, 2, 3])
        jd.setattr("dims", ["x", "y"])
        jd.setattr("units", "mm")
        self.assertEqual(jd.getattr("dims"), ["x", "y"])
        self.assertEqual(jd.getattr("units"), "mm")

    def test_second_level_attr(self):
        jd = jdict({"a": [1, 2, 3], "b": [4, 5, 6]})
        jd.a.setattr("dims", ["x", "y"])
        self.assertEqual(jd.a.getattr("dims"), ["x", "y"])

    def test_attr_independence(self):
        jd = jdict({"a": [1, 2], "b": [3, 4]})
        jd.a.setattr("dims", ["x", "y"])
        jd.b.setattr("dims", ["rows", "cols"])
        self.assertEqual(jd.a.getattr("dims"), ["x", "y"])
        self.assertEqual(jd.b.getattr("dims"), ["rows", "cols"])

    def test_attr_overwrite(self):
        jd = jdict([1, 2, 3])
        jd.setattr("dims", ["old1", "old2"])
        jd.setattr("dims", ["new1", "new2"])
        self.assertEqual(jd.getattr("dims"), ["new1", "new2"])

    def test_nonexistent_attr(self):
        jd = jdict([1, 2, 3])
        self.assertIsNone(jd.getattr("nonexistent"))

    def test_curly_bracket_attr(self):
        jd = jdict([1, 2, 3])
        jd["{dims}"] = ["x", "y", "z"]
        self.assertEqual(jd["{dims}"], ["x", "y", "z"])


class TestJdictDims(unittest.TestCase):
    """Test dimension-based indexing without coords"""

    def test_root_level_dim_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        result = jd.x(1)()
        np.testing.assert_array_equal(result, [4, 5, 6, 7])

    def test_root_level_dim_select_second_dim(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        result = jd.y(2)()
        np.testing.assert_array_equal(result, [2, 6, 10])  # NumPy drops singleton dim

    def test_second_level_dim_select(self):
        jd = jdict({"data": np.arange(12).reshape(3, 4)})
        jd.data.setattr("dims", ["row", "col"])
        result = jd.data.row(0)()
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_third_level_dim_select(self):
        jd = jdict({"level1": {"level2": {"arr": np.arange(6).reshape(2, 3)}}})
        jd.level1.level2.arr.setattr("dims", ["i", "j"])
        result = jd.level1.level2.arr.i(1)()
        np.testing.assert_array_equal(result, [3, 4, 5])

    def test_dim_slice_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        result = jd.x(slice(0, 2))()
        np.testing.assert_array_equal(result, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_dim_list_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        result = jd.x([0, 2])()
        np.testing.assert_array_equal(result, [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_3d_dim_select(self):
        jd = jdict(np.arange(24).reshape(2, 3, 4))
        jd.setattr("dims", ["x", "y", "z"])
        result = jd.y(1)()
        expected = np.arange(24).reshape(2, 3, 4)[:, 1, :]  # scalar index drops dim
        np.testing.assert_array_equal(result, expected)

    def test_sibling_dims(self):
        jd = jdict(
            {"exp1": np.arange(6).reshape(2, 3), "exp2": np.arange(8).reshape(2, 4)}
        )
        jd.exp1.setattr("dims", ["t", "s"])
        jd.exp2.setattr("dims", ["x", "y"])
        result1 = jd.exp1.t(1)()
        result2 = jd.exp2.x(0)()
        np.testing.assert_array_equal(result1, [3, 4, 5])
        np.testing.assert_array_equal(result2, [0, 1, 2, 3])


class TestJdictCoords(unittest.TestCase):
    """Test coordinate-based indexing"""

    def test_string_coord_single_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.x("b")()
        np.testing.assert_array_equal(result, [4, 5, 6, 7])

    def test_string_coord_single_select_second_dim(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.y("r")()
        np.testing.assert_array_equal(result, [2, 6, 10])  # NumPy drops singleton dim

    def test_numeric_coord_single_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": [10, 20, 30], "y": [100, 200, 300, 400]})
        result = jd.x(20)()
        np.testing.assert_array_equal(result, [4, 5, 6, 7])

    def test_numeric_coord_single_select_second_dim(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": [10, 20, 30], "y": [100, 200, 300, 400]})
        result = jd.y(300)()
        np.testing.assert_array_equal(result, [2, 6, 10])  # NumPy drops singleton dim

    def test_string_coord_multi_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.x(["a", "c"])()
        np.testing.assert_array_equal(result, [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_numeric_coord_multi_select(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": [10, 20, 30], "y": [100, 200, 300, 400]})
        result = jd.x([10, 30])()
        np.testing.assert_array_equal(result, [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_string_coord_slice(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.x({"start": "a", "stop": "b"})()
        np.testing.assert_array_equal(result, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_direct_index_with_string_coords(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.x(1)()  # direct index, not coord lookup
        np.testing.assert_array_equal(result, [4, 5, 6, 7])

    def test_dims_without_coords(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        result = jd.x(1)()
        np.testing.assert_array_equal(result, [4, 5, 6, 7])

    def test_second_level_coords(self):
        jd = jdict({"data": np.arange(12).reshape(3, 4)})
        jd.data.setattr("dims", ["row", "col"])
        jd.data.setattr("coords", {"row": ["r1", "r2", "r3"], "col": [1, 2, 3, 4]})
        result_str = jd.data.row("r2")()
        result_num = jd.data.col(3)()
        np.testing.assert_array_equal(result_str, [4, 5, 6, 7])
        np.testing.assert_array_equal(
            result_num, [2, 6, 10]
        )  # NumPy drops singleton dim

    def test_third_level_coords(self):
        jd = jdict({"level1": {"level2": {"arr": np.arange(6).reshape(2, 3)}}})
        jd.level1.level2.arr.setattr("dims", ["i", "j"])
        jd.level1.level2.arr.setattr("coords", {"i": ["x", "y"], "j": ["a", "b", "c"]})
        result_i = jd.level1.level2.arr.i("y")()
        result_j = jd.level1.level2.arr.j("b")()
        np.testing.assert_array_equal(result_i, [3, 4, 5])
        np.testing.assert_array_equal(result_j, [1, 4])  # NumPy drops singleton dim

    def test_sibling_coords(self):
        jd = jdict(
            {"exp1": np.arange(6).reshape(2, 3), "exp2": np.arange(8).reshape(2, 4)}
        )
        jd.exp1.setattr("dims", ["t", "s"])
        jd.exp1.setattr("coords", {"t": ["t1", "t2"], "s": ["s1", "s2", "s3"]})
        jd.exp2.setattr("dims", ["x", "y"])
        jd.exp2.setattr("coords", {"x": [0, 1], "y": [10, 20, 30, 40]})
        result1 = jd.exp1.t("t2")()
        result2 = jd.exp2.y(30)()
        np.testing.assert_array_equal(result1, [3, 4, 5])
        np.testing.assert_array_equal(result2, [2, 6])  # NumPy drops singleton dim

    def test_3d_coords(self):
        jd = jdict(np.arange(24).reshape(2, 3, 4))
        jd.setattr("dims", ["x", "y", "z"])
        jd.setattr(
            "coords", {"x": ["a", "b"], "y": [10, 20, 30], "z": ["p", "q", "r", "s"]}
        )
        result_x = jd.x("b")()
        result_y = jd.y(20)()
        result_z = jd.z("r")()
        expected_x = np.arange(24).reshape(2, 3, 4)[1, :, :]  # scalar drops dim
        expected_y = np.arange(24).reshape(2, 3, 4)[:, 1, :]
        expected_z = np.arange(24).reshape(2, 3, 4)[:, :, 2]
        np.testing.assert_array_equal(result_x, expected_x)
        np.testing.assert_array_equal(result_y, expected_y)
        np.testing.assert_array_equal(result_z, expected_z)

    def test_getattr_returns_coords(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": [1, 2, 3, 4]})
        coords = jd.getattr("coords")
        self.assertEqual(coords, {"x": ["a", "b", "c"], "y": [1, 2, 3, 4]})

    def test_coord_not_found_raises(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        with self.assertRaises(ValueError):
            jd.x("z")()


class TestJdictCoordsCascade(unittest.TestCase):
    """Test cascaded coordinate selections - requires updated _DimAccessor that tracks reduced dims"""

    def test_cascade_two_singles(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        # After jd.x("b"), result is 1D array [4,5,6,7], dims should be ["y"]
        result = jd.x("b").y("r")()
        self.assertEqual(result, 6)

    def test_cascade_reverse_order(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.y("q").x("c")()
        self.assertEqual(result, 9)

    def test_cascade_multi_then_single(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.x(["a", "c"]).y("s")()
        np.testing.assert_array_equal(result, [3, 11])  # NumPy drops singleton dim

    def test_cascade_single_then_multi(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": ["a", "b", "c"], "y": ["p", "q", "r", "s"]})
        result = jd.y("p").x(["a", "b"])()
        np.testing.assert_array_equal(result, [0, 4])

    def test_cascade_numeric_coords(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": [10, 20, 30], "y": [100, 200, 300, 400]})
        result = jd.x(20).y(300)()
        self.assertEqual(result, 6)

    def test_cascade_numeric_multi_then_single(self):
        jd = jdict(np.arange(12).reshape(3, 4))
        jd.setattr("dims", ["x", "y"])
        jd.setattr("coords", {"x": [10, 20, 30], "y": [100, 200, 300, 400]})
        result = jd.x([10, 30]).y(400)()
        np.testing.assert_array_equal(result, [3, 11])

    def test_cascade_in_nested_struct(self):
        jd = jdict({"data": np.arange(12).reshape(3, 4)})
        jd.data.setattr("dims", ["row", "col"])
        jd.data.setattr(
            "coords", {"row": ["r1", "r2", "r3"], "col": ["c1", "c2", "c3", "c4"]}
        )
        result = jd.data.row("r2").col("c3")()
        self.assertEqual(result, 6)

    def test_cascade_3d_two_dims(self):
        jd = jdict(np.arange(24).reshape(2, 3, 4))
        jd.setattr("dims", ["x", "y", "z"])
        jd.setattr(
            "coords", {"x": ["a", "b"], "y": ["p", "q", "r"], "z": ["i", "j", "k", "l"]}
        )
        result = jd.x("b").z("k")()
        np.testing.assert_array_equal(result, [14, 18, 22])

    def test_cascade_3d_all_three_dims(self):
        jd = jdict(np.arange(24).reshape(2, 3, 4))
        jd.setattr("dims", ["x", "y", "z"])
        jd.setattr(
            "coords", {"x": ["a", "b"], "y": ["p", "q", "r"], "z": ["i", "j", "k", "l"]}
        )
        result = jd.x("a").y("q").z("j")()
        self.assertEqual(result, 5)


class TestJdictDeepNesting(unittest.TestCase):
    """Test dims/coords at deeply nested levels"""

    def test_four_levels_deep_dims(self):
        jd = jdict({"a": {"b": {"c": {"d": np.arange(6).reshape(2, 3)}}}})
        jd.a.b.c.d.setattr("dims", ["x", "y"])
        result = jd.a.b.c.d.x(1)()
        np.testing.assert_array_equal(result, [3, 4, 5])

    def test_four_levels_deep_coords(self):
        jd = jdict({"a": {"b": {"c": {"d": np.arange(6).reshape(2, 3)}}}})
        jd.a.b.c.d.setattr("dims", ["x", "y"])
        jd.a.b.c.d.setattr("coords", {"x": ["r1", "r2"], "y": ["c1", "c2", "c3"]})
        result = jd.a.b.c.d.x("r2").y("c2")()
        self.assertEqual(result, 4)

    def test_multiple_arrays_deep(self):
        jd = jdict(
            {
                "experiment": {
                    "trial1": {"data": np.arange(12).reshape(3, 4)},
                    "trial2": {"data": np.arange(8).reshape(2, 4)},
                }
            }
        )
        jd.experiment.trial1.data.setattr("dims", ["time", "channel"])
        jd.experiment.trial1.data.setattr(
            "coords",
            {"time": ["t1", "t2", "t3"], "channel": ["ch1", "ch2", "ch3", "ch4"]},
        )
        jd.experiment.trial2.data.setattr("dims", ["time", "channel"])
        jd.experiment.trial2.data.setattr(
            "coords", {"time": ["t1", "t2"], "channel": ["ch1", "ch2", "ch3", "ch4"]}
        )

        result1 = jd.experiment.trial1.data.time("t2").channel("ch3")()
        result2 = jd.experiment.trial2.data.time("t1").channel("ch4")()
        self.assertEqual(result1, 6)
        self.assertEqual(result2, 3)

    def test_mixed_depth_attrs(self):
        jd = jdict(
            {
                "shallow": np.arange(4).reshape(2, 2),
                "deep": {"nested": {"arr": np.arange(6).reshape(2, 3)}},
            }
        )
        jd.shallow.setattr("dims", ["a", "b"])
        jd.shallow.setattr("coords", {"a": ["x", "y"], "b": ["p", "q"]})
        jd.deep.nested.arr.setattr("dims", ["i", "j"])
        jd.deep.nested.arr.setattr("coords", {"i": [0, 1], "j": [10, 20, 30]})

        result_shallow = jd.shallow.a("y").b("p")()
        result_deep = jd.deep.nested.arr.i(1).j(20)()
        self.assertEqual(result_shallow, 2)
        self.assertEqual(result_deep, 4)


class TestJsonSchema(unittest.TestCase):
    def test_setschema_from_dict(self):
        jd = jdict({"name": "John", "age": 30})
        jd.setschema({"type": "object"})
        self.assertIsNotNone(jd.getschema())

    def test_getschema_as_json(self):
        jd = jdict({"name": "John"})
        jd.setschema({"type": "object"})
        js = jd.getschema("json")
        self.assertIn("object", js)

    def test_clear_schema(self):
        jd = jdict({"x": 1})
        jd.setschema({"type": "object"})
        jd.setschema(None)
        self.assertIsNone(jd.getschema())

    def test_validate_string_pass(self):
        jd = jdict("hello")
        jd.setschema({"type": "string"})
        self.assertEqual(jd.validate(), [])

    def test_validate_string_fail(self):
        jd = jdict(123)
        jd.setschema({"type": "string"})
        self.assertNotEqual(jd.validate(), [])

    def test_validate_integer_pass(self):
        jd = jdict(42)
        jd.setschema({"type": "integer"})
        self.assertEqual(jd.validate(), [])

    def test_validate_integer_fail(self):
        jd = jdict(3.14)
        jd.setschema({"type": "integer"})
        self.assertNotEqual(jd.validate(), [])

    def test_validate_number_pass(self):
        jd = jdict(3.14)
        jd.setschema({"type": "number"})
        self.assertEqual(jd.validate(), [])

    def test_validate_boolean_pass(self):
        jd = jdict(True)
        jd.setschema({"type": "boolean"})
        self.assertEqual(jd.validate(), [])

    def test_validate_boolean_fail(self):
        jd = jdict(1)
        jd.setschema({"type": "boolean"})
        self.assertNotEqual(jd.validate(), [])

    def test_validate_null_pass(self):
        jd = jdict(None)
        jd.setschema({"type": "null"})
        self.assertEqual(jd.validate(), [])

    def test_validate_array_pass(self):
        jd = jdict([1, 2])
        jd.setschema({"type": "array"})
        self.assertEqual(jd.validate(), [])

    def test_validate_object_pass(self):
        jd = jdict({"a": 1})
        jd.setschema({"type": "object"})
        self.assertEqual(jd.validate(), [])


class TestSchemaNumeric(unittest.TestCase):
    def test_minimum_pass(self):
        jd = jdict(10)
        jd.setschema({"type": "integer", "minimum": 5})
        self.assertEqual(jd.validate(), [])

    def test_minimum_fail(self):
        jd = jdict(3)
        jd.setschema({"type": "integer", "minimum": 5})
        self.assertNotEqual(jd.validate(), [])

    def test_maximum_pass(self):
        jd = jdict(5)
        jd.setschema({"type": "integer", "maximum": 10})
        self.assertEqual(jd.validate(), [])

    def test_maximum_fail(self):
        jd = jdict(15)
        jd.setschema({"type": "integer", "maximum": 10})
        self.assertNotEqual(jd.validate(), [])

    def test_exclusive_minimum_pass(self):
        jd = jdict(6)
        jd.setschema({"type": "integer", "exclusiveMinimum": 5})
        self.assertEqual(jd.validate(), [])

    def test_exclusive_minimum_fail(self):
        jd = jdict(5)
        jd.setschema({"type": "integer", "exclusiveMinimum": 5})
        self.assertNotEqual(jd.validate(), [])

    def test_exclusive_maximum_pass(self):
        jd = jdict(4)
        jd.setschema({"type": "integer", "exclusiveMaximum": 5})
        self.assertEqual(jd.validate(), [])

    def test_exclusive_maximum_fail(self):
        jd = jdict(5)
        jd.setschema({"type": "integer", "exclusiveMaximum": 5})
        self.assertNotEqual(jd.validate(), [])

    def test_multiple_of_pass(self):
        jd = jdict(15)
        jd.setschema({"type": "integer", "multipleOf": 5})
        self.assertEqual(jd.validate(), [])

    def test_multiple_of_fail(self):
        jd = jdict(17)
        jd.setschema({"type": "integer", "multipleOf": 5})
        self.assertNotEqual(jd.validate(), [])


class TestSchemaString(unittest.TestCase):
    def test_min_length_pass(self):
        jd = jdict("hello")
        jd.setschema({"type": "string", "minLength": 3})
        self.assertEqual(jd.validate(), [])

    def test_min_length_fail(self):
        jd = jdict("hi")
        jd.setschema({"type": "string", "minLength": 3})
        self.assertNotEqual(jd.validate(), [])

    def test_max_length_pass(self):
        jd = jdict("hi")
        jd.setschema({"type": "string", "maxLength": 5})
        self.assertEqual(jd.validate(), [])

    def test_max_length_fail(self):
        jd = jdict("hello world")
        jd.setschema({"type": "string", "maxLength": 5})
        self.assertNotEqual(jd.validate(), [])

    def test_pattern_pass(self):
        jd = jdict("abc123")
        jd.setschema({"type": "string", "pattern": "^[a-z]+[0-9]+$"})
        self.assertEqual(jd.validate(), [])

    def test_pattern_fail(self):
        jd = jdict("123abc")
        jd.setschema({"type": "string", "pattern": "^[a-z]+[0-9]+$"})
        self.assertNotEqual(jd.validate(), [])

    def test_format_email_pass(self):
        jd = jdict("user@example.com")
        jd.setschema({"type": "string", "format": "email"})
        self.assertEqual(jd.validate(), [])

    def test_format_email_fail(self):
        jd = jdict("notanemail")
        jd.setschema({"type": "string", "format": "email"})
        self.assertNotEqual(jd.validate(), [])


class TestSchemaEnumConst(unittest.TestCase):
    def test_enum_pass(self):
        jd = jdict("red")
        jd.setschema({"enum": ["red", "green", "blue"]})
        self.assertEqual(jd.validate(), [])

    def test_enum_fail(self):
        jd = jdict("yellow")
        jd.setschema({"enum": ["red", "green", "blue"]})
        self.assertNotEqual(jd.validate(), [])

    def test_const_pass(self):
        jd = jdict("fixed")
        jd.setschema({"const": "fixed"})
        self.assertEqual(jd.validate(), [])

    def test_const_fail(self):
        jd = jdict("other")
        jd.setschema({"const": "fixed"})
        self.assertNotEqual(jd.validate(), [])


class TestSchemaArray(unittest.TestCase):
    def test_min_items_pass(self):
        jd = jdict([1, 2, 3])
        jd.setschema({"type": "array", "minItems": 2})
        self.assertEqual(jd.validate(), [])

    def test_min_items_fail(self):
        jd = jdict([1])
        jd.setschema({"type": "array", "minItems": 2})
        self.assertNotEqual(jd.validate(), [])

    def test_max_items_pass(self):
        jd = jdict([1, 2])
        jd.setschema({"type": "array", "maxItems": 3})
        self.assertEqual(jd.validate(), [])

    def test_max_items_fail(self):
        jd = jdict([1, 2, 3, 4])
        jd.setschema({"type": "array", "maxItems": 3})
        self.assertNotEqual(jd.validate(), [])

    def test_unique_items_pass(self):
        jd = jdict([1, 2, 3])
        jd.setschema({"type": "array", "uniqueItems": True})
        self.assertEqual(jd.validate(), [])

    def test_unique_items_fail(self):
        jd = jdict([1, 2, 2])
        jd.setschema({"type": "array", "uniqueItems": True})
        self.assertNotEqual(jd.validate(), [])

    def test_items_pass(self):
        jd = jdict([1, 2, 3])
        jd.setschema({"type": "array", "items": {"type": "integer"}})
        self.assertEqual(jd.validate(), [])

    def test_items_fail(self):
        jd = jdict([1, "two", 3])
        jd.setschema({"type": "array", "items": {"type": "integer"}})
        self.assertNotEqual(jd.validate(), [])

    def test_contains_pass(self):
        jd = jdict([1, "hello", 3])
        jd.setschema({"type": "array", "contains": {"type": "string"}})
        self.assertEqual(jd.validate(), [])

    def test_contains_fail(self):
        jd = jdict([1, 2, 3])
        jd.setschema({"type": "array", "contains": {"type": "string"}})
        self.assertNotEqual(jd.validate(), [])


class TestSchemaObject(unittest.TestCase):
    def test_required_pass(self):
        jd = jdict({"name": "John", "age": 30})
        jd.setschema({"type": "object", "required": ["name", "age"]})
        self.assertEqual(jd.validate(), [])

    def test_required_fail(self):
        jd = jdict({"name": "John"})
        jd.setschema({"type": "object", "required": ["name", "age"]})
        self.assertNotEqual(jd.validate(), [])

    def test_properties_pass(self):
        jd = jdict({"name": "John", "age": 30})
        jd.setschema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            }
        )
        self.assertEqual(jd.validate(), [])

    def test_properties_fail(self):
        jd = jdict({"name": 123, "age": 30})
        jd.setschema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            }
        )
        self.assertNotEqual(jd.validate(), [])

    def test_min_properties_pass(self):
        jd = jdict({"a": 1, "b": 2})
        jd.setschema({"type": "object", "minProperties": 2})
        self.assertEqual(jd.validate(), [])

    def test_min_properties_fail(self):
        jd = jdict({"a": 1})
        jd.setschema({"type": "object", "minProperties": 2})
        self.assertNotEqual(jd.validate(), [])

    def test_additional_properties_pass(self):
        jd = jdict({"name": "John"})
        jd.setschema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        self.assertEqual(jd.validate(), [])

    def test_additional_properties_fail(self):
        jd = jdict({"name": "John", "extra": "field"})
        jd.setschema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        self.assertNotEqual(jd.validate(), [])


class TestSchemaComposition(unittest.TestCase):
    def test_all_of_pass(self):
        jd = jdict(10)
        jd.setschema({"allOf": [{"type": "integer"}, {"minimum": 5}]})
        self.assertEqual(jd.validate(), [])

    def test_all_of_fail(self):
        jd = jdict(3)
        jd.setschema({"allOf": [{"type": "integer"}, {"minimum": 5}]})
        self.assertNotEqual(jd.validate(), [])

    def test_any_of_pass(self):
        jd = jdict("hello")
        jd.setschema({"anyOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertEqual(jd.validate(), [])

    def test_any_of_fail(self):
        jd = jdict(True)
        jd.setschema({"anyOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertNotEqual(jd.validate(), [])

    def test_one_of_pass(self):
        jd = jdict(5)
        jd.setschema({"oneOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertEqual(jd.validate(), [])

    def test_one_of_fail(self):
        jd = jdict(True)
        jd.setschema({"oneOf": [{"type": "integer"}, {"type": "string"}]})
        self.assertNotEqual(jd.validate(), [])

    def test_not_pass(self):
        jd = jdict("hello")
        jd.setschema({"not": {"type": "integer"}})
        self.assertEqual(jd.validate(), [])

    def test_not_fail(self):
        jd = jdict(42)
        jd.setschema({"not": {"type": "integer"}})
        self.assertNotEqual(jd.validate(), [])


class TestSchemaValidatedAssignment(unittest.TestCase):
    def test_le_root_string_pass(self):
        jd = jdict("hello")
        jd.setschema({"type": "string"})
        jd <= "world"
        self.assertEqual(jd(), "world")

    def test_le_root_integer_pass(self):
        jd = jdict(10)
        jd.setschema({"type": "integer", "minimum": 0, "maximum": 100})
        jd <= 50
        self.assertEqual(jd(), 50)

    def test_le_root_integer_fail(self):
        jd = jdict(10)
        jd.setschema({"type": "integer", "minimum": 0})
        with self.assertRaises(ValueError):
            jd <= -5

    def test_le_subkey_pass(self):
        jd = jdict({"name": "John", "age": 30})
        jd.setschema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                },
            }
        )
        jd.name <= "Jane"
        self.assertEqual(jd().get("name"), "Jane")

    def test_le_subkey_fail(self):
        jd = jdict({"name": "John", "age": 30})
        jd.setschema(
            {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}}
        )
        with self.assertRaises(ValueError):
            jd.age <= -10


class TestSchemaGeneration(unittest.TestCase):
    def test_generate_null(self):
        result = jsonschema({"type": "null"}, generate="all")
        self.assertIsNone(result)

    def test_generate_boolean(self):
        result = jsonschema({"type": "boolean"}, generate="all")
        self.assertEqual(result, False)

    def test_generate_integer(self):
        result = jsonschema({"type": "integer"}, generate="all")
        self.assertEqual(result, 0)

    def test_generate_number(self):
        result = jsonschema({"type": "number"}, generate="all")
        self.assertEqual(result, 0.0)

    def test_generate_string(self):
        result = jsonschema({"type": "string"}, generate="all")
        self.assertEqual(result, "")

    def test_generate_with_default(self):
        result = jsonschema({"type": "string", "default": "hello"}, generate="all")
        self.assertEqual(result, "hello")

    def test_generate_with_const(self):
        result = jsonschema({"const": "fixed"}, generate="all")
        self.assertEqual(result, "fixed")

    def test_generate_with_enum(self):
        result = jsonschema({"enum": ["red", "green", "blue"]}, generate="all")
        self.assertEqual(result, "red")

    def test_generate_int_with_minimum(self):
        result = jsonschema({"type": "integer", "minimum": 10}, generate="all")
        self.assertEqual(result, 10)

    def test_generate_int_with_exclusive_min(self):
        result = jsonschema({"type": "integer", "exclusiveMinimum": 10}, generate="all")
        self.assertEqual(result, 11)

    def test_generate_string_min_length(self):
        result = jsonschema({"type": "string", "minLength": 5}, generate="all")
        self.assertEqual(len(result), 5)

    def test_generate_email_format(self):
        result = jsonschema({"type": "string", "format": "email"}, generate="all")
        self.assertIn("@", result)

    def test_generate_empty_array(self):
        result = jsonschema({"type": "array"}, generate="all")
        self.assertEqual(result, [])

    def test_generate_array_min_items(self):
        result = jsonschema(
            {"type": "array", "minItems": 3, "items": {"type": "integer"}},
            generate="all",
        )
        self.assertEqual(len(result), 3)

    def test_generate_empty_object(self):
        result = jsonschema({"type": "object"}, generate="all")
        self.assertEqual(result, {})

    def test_generate_object_required(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string", "default": "John"},
                "age": {"type": "integer"},
                "optional": {"type": "string"},
            },
        }
        result = jsonschema(schema, generate="all")
        self.assertIn("name", result)
        self.assertIn("age", result)
        self.assertEqual(result["name"], "John")


class TestKind(unittest.TestCase):
    def test_kind_date_schema_set(self):
        jd = jdict(None, kind="date")
        self.assertIsNotNone(jd.getschema())

    def test_kind_date_attr_set(self):
        jd = jdict(None, kind="date")
        self.assertEqual(jd.getattr("$", "kind"), "date")

    def test_kind_uuid_schema_set(self):
        jd = jdict(None, kind="uuid")
        self.assertIsNotNone(jd.getschema())

    def test_kind_time_schema_set(self):
        jd = jdict(None, kind="time")
        self.assertIsNotNone(jd.getschema())

    def test_kind_unknown_fail(self):
        with self.assertRaises(ValueError):
            jdict(None, kind="unknownkind")

    def test_kind_date_valid_assign(self):
        jd = jdict(None, kind="date")
        jd.year = 2025
        jd.month = 1
        jd.day = 12
        self.assertEqual(jd(), "2025-01-12")

    def test_kind_date_month_max_fail(self):
        jd = jdict(None, kind="date")
        with self.assertRaises(ValueError):
            jd.month = 13

    def test_kind_date_month_min_fail(self):
        jd = jdict(None, kind="date")
        with self.assertRaises(ValueError):
            jd.month = 0

    def test_kind_time_valid_assign(self):
        jd = jdict(None, kind="time")
        jd.hour = 14
        jd.min = 30
        jd.sec = 45.5
        self.assertEqual(jd(), "14:30:45.500")

    def test_kind_time_hour_max_fail(self):
        jd = jdict(None, kind="time")
        with self.assertRaises(ValueError):
            jd.hour = 24

    def test_kind_uuid_format(self):
        jd = jdict(
            {
                "time_low": 1427104768,
                "time_mid": 58011,
                "time_high": 16852,
                "clock_seq": 42774,
                "node": 75334816473088,
            },
            kind="uuid",
        )
        self.assertEqual(jd(), "550fe400-e29b-41d4-a716-448440f9a000")

    def test_kind_email_format(self):
        jd = jdict({"user": "john.doe", "domain": "example.com"}, kind="email")
        self.assertEqual(jd(), "john.doe@example.com")

    def test_kind_uri_format_basic(self):
        jd = jdict({"scheme": "https", "host": "example.com"}, kind="uri")
        self.assertEqual(jd(), "https://example.com")

    def test_kind_uri_format_port(self):
        jd = jdict({"scheme": "https", "host": "example.com", "port": 443}, kind="uri")
        self.assertEqual(jd(), "https://example.com:443")


@unittest.skipIf(False, "NumPy required")
class TestBinaryValidation(unittest.TestCase):
    def test_bintype_uint8_pass(self):
        jd = jdict(np.array([1, 2, 3], dtype=np.uint8))
        jd.setschema({"binType": "uint8"})
        self.assertEqual(jd.validate(), [])

    def test_bintype_uint8_fail(self):
        jd = jdict(np.array([1, 2, 3], dtype=np.float64))
        jd.setschema({"binType": "uint8"})
        self.assertNotEqual(jd.validate(), [])

    def test_bintype_int32_pass(self):
        jd = jdict(np.array([-100, 0, 100], dtype=np.int32))
        jd.setschema({"binType": "int32"})
        self.assertEqual(jd.validate(), [])

    def test_bintype_logical_pass(self):
        jd = jdict(np.array([[True, False], [False, True]], dtype=np.bool_))
        jd.setschema({"binType": "logical"})
        self.assertEqual(jd.validate(), [])

    def test_min_dims_pass(self):
        jd = jdict(np.zeros((3, 4)))
        jd.setschema({"minDims": [2, 3]})
        self.assertEqual(jd.validate(), [])

    def test_min_dims_fail(self):
        jd = jdict(np.zeros((1, 4)))
        jd.setschema({"minDims": [2, 3]})
        self.assertNotEqual(jd.validate(), [])

    def test_max_dims_pass(self):
        jd = jdict(np.zeros((5, 5)))
        jd.setschema({"maxDims": [10, 10]})
        self.assertEqual(jd.validate(), [])

    def test_max_dims_fail(self):
        jd = jdict(np.zeros((15, 5)))
        jd.setschema({"maxDims": [10, 10]})
        self.assertNotEqual(jd.validate(), [])

    def test_min_dims_vector_pass(self):
        jd = jdict(np.arange(10))
        jd.setschema({"minDims": [5]})
        self.assertEqual(jd.validate(), [])

    def test_min_dims_vector_fail(self):
        jd = jdict(np.arange(3))
        jd.setschema({"minDims": [5]})
        self.assertNotEqual(jd.validate(), [])

    def test_combined_bintype_dims_pass(self):
        jd = jdict(np.zeros((3, 4), dtype=np.uint8))
        jd.setschema({"binType": "uint8", "minDims": [2, 2], "maxDims": [10, 10]})
        self.assertEqual(jd.validate(), [])

    def test_combined_bintype_dims_fail_type(self):
        jd = jdict(np.zeros((3, 4), dtype=np.float64))
        jd.setschema({"binType": "uint8", "minDims": [2, 2], "maxDims": [10, 10]})
        self.assertNotEqual(jd.validate(), [])


class TestAttr2Schema(unittest.TestCase):
    def test_attr2schema_basic(self):
        jd = jdict({"age": 25})
        jd.age.setattr(":type", "integer")
        jd.age.setattr(":minimum", 0)
        jd.age.setattr(":maximum", 150)
        schema = jd.attr2schema(title="Test Schema")
        self.assertIn("properties", schema)
        self.assertIn("age", schema["properties"])
        self.assertEqual(schema["title"], "Test Schema")

    def test_attr2schema_roundtrip(self):
        jd = jdict({"age": 25})
        jd.age.setattr(":type", "integer")
        jd.age.setattr(":minimum", 0)
        schema = jd.attr2schema()
        jd.setschema(schema)
        self.assertEqual(jd.validate(), [])

    def test_attr2schema_multi_field(self):
        jd = jdict({"name": "test", "count": 5})
        jd.name.setattr(":type", "string")
        jd.name.setattr(":minLength", 1)
        jd.count.setattr(":type", "integer")
        jd.count.setattr(":minimum", 0)
        schema = jd.attr2schema()
        self.assertIn("name", schema["properties"])
        self.assertIn("count", schema["properties"])


if __name__ == "__main__":
    unittest.main()
