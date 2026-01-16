"""
jdict - A universal dictionary-like interface with JSONPath support, attributes, and JSON Schema validation

Author: Qianqian Fang (q.fang <at> neu.edu)
Converted from MATLAB by Claude

Example:
    jd = jdict()
    jd.key1 = {'subkey1': 1, 'subkey2': [1, 2, 3]}
    jd.key2 = 'str'
    jd['$.key1.subkey1']  # JSONPath access
    jd.key1.subkey1()     # returns 1
    jd.tojson()           # convert to JSON string
"""

__all__ = [
    "jdict",
]

##====================================================================================
## dependent libraries
##====================================================================================

import json
import re
import copy
from typing import Any, Optional, Union, List, Dict
from urllib.request import urlopen

import numpy as np

from .jschema import jsonschema
from .jpath import jsonpath


class jdict:
    """A universal dictionary-like interface with JSONPath support and schema validation."""

    __slots__ = ("_data", "_attr", "_schema", "_currentpath", "_root", "_flags")

    def __init__(self, data=None, **kwargs):
        object.__setattr__(self, "_data", None)
        object.__setattr__(self, "_attr", {})
        object.__setattr__(self, "_schema", None)
        object.__setattr__(self, "_currentpath", "$")
        object.__setattr__(self, "_root", self)
        object.__setattr__(self, "_flags", kwargs)

        if "attr" in kwargs:
            object.__setattr__(self, "_attr", kwargs["attr"])
        if "schema" in kwargs:
            self.setschema(kwargs["schema"])

        kindval = kwargs.get("kind", "")

        if data is not None:
            if isinstance(data, str) and re.match(r"^https?://", data, re.I):
                try:
                    from jdata import load as jd_load

                    object.__setattr__(self, "_data", jd_load(data))
                except:
                    try:
                        with urlopen(data) as resp:
                            object.__setattr__(
                                self, "_data", json.loads(resp.read().decode())
                            )
                    except:
                        object.__setattr__(self, "_data", data)
            elif isinstance(data, jdict):
                object.__setattr__(self, "_data", copy.deepcopy(data._data))
                object.__setattr__(self, "_attr", copy.deepcopy(data._attr))
                self.setschema(copy.deepcopy(data._schema))
                object.__setattr__(self, "_currentpath", data._currentpath)
                object.__setattr__(self, "_flags", copy.deepcopy(data._flags))
            else:
                object.__setattr__(self, "_data", data)

        if kindval:
            kindschema = _getkindschema(kindval)
            if kindschema:
                self.setschema(kindschema)
            elif not self._schema:
                raise ValueError(
                    f'Unknown kind "{kindval}". Use: uuid, date, time, datetime, email, uri'
                )
            self.setattr("$", "kind", kindval)
            if self._data is None:
                generated = jsonschema(kindschema, generate="all")
                object.__setattr__(self, "_data", generated)

    def __call__(self, *args):
        if args:
            return self.v(*args)
        kindval = self.getattr("$", "kind")
        if kindval and isinstance(self._data, dict):
            formatted = _formatkind(kindval, self._data)
            if formatted is not None:
                return formatted
        return self._data

    def v(self, *args):
        if not args:
            return self._data
        idx = args[0]
        data = self._data
        if isinstance(data, (list, tuple)):
            if isinstance(idx, slice):
                return data[idx]
            elif isinstance(idx, (list, tuple)):
                return [data[i] for i in idx]
            return data[idx]
        elif isinstance(data, dict):
            return data[idx]
        elif hasattr(data, "__getitem__"):
            return data[idx]
        return data

    def __getattr__(self, name):
        if name.startswith("_") or name in self.__slots__:
            return object.__getattribute__(self, name)

        data = object.__getattribute__(self, "_data")
        currentpath = object.__getattribute__(self, "_currentpath")
        attr = object.__getattribute__(self, "_attr")
        schema = object.__getattribute__(self, "_schema")
        root = object.__getattribute__(self, "_root")

        methods = (
            "tojson",
            "fromjson",
            "keys",
            "len",
            "size",
            "iskey",
            "isfield",
            "rmfield",
            "setattr",
            "getattr",
            "setschema",
            "getschema",
            "validate",
            "attr2schema",
            "v",
        )
        if name in methods:
            return object.__getattribute__(self, name)

        # Check for dimension-based indexing
        dims = _get_attr_value(attr, currentpath, "dims")
        if dims is not None and isinstance(dims, (list, tuple)) and name in dims:
            return _DimAccessor(self, name)

        if data is None:
            val = None
        elif isinstance(data, dict):
            val = data.get(name)
        elif isinstance(data, (list, tuple)):
            if name.isdigit():
                idx = int(name)
                val = data[idx] if idx < len(data) else None
            else:
                val = None
        elif hasattr(data, name):
            val = getattr(data, name, None)
        else:
            val = None

        escapedkey = _esckey(name)
        newpath = f"{currentpath}.{escapedkey}"

        newobj = jdict.__new__(jdict)
        object.__setattr__(newobj, "_data", val)
        object.__setattr__(newobj, "_attr", attr)
        object.__setattr__(newobj, "_schema", schema)
        object.__setattr__(newobj, "_currentpath", newpath)
        object.__setattr__(newobj, "_root", root)
        object.__setattr__(newobj, "_flags", {})
        return newobj

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self.__slots__:
            object.__setattr__(self, name, value)
            return

        data = object.__getattribute__(self, "_data")
        currentpath = object.__getattribute__(self, "_currentpath")
        schema = object.__getattribute__(self, "_schema")
        root = object.__getattribute__(self, "_root")
        attr = object.__getattribute__(self, "_attr")

        kindval = _get_attr_value(attr, "$", "kind")
        if schema and kindval:
            targetpath = f"{currentpath}.{_esckey(name)}"
            subschema = jsonschema(schema, None, getsubschema=targetpath)
            if subschema:
                valid, errs = jsonschema(value, subschema, rootschema=schema)
                if not valid:
                    raise ValueError(
                        f'Schema validation failed for "{targetpath}": {"; ".join(errs)}'
                    )

        if data is None:
            data = {}
            object.__setattr__(self, "_data", data)

        if isinstance(data, dict):
            data[name] = value
        elif isinstance(data, list):
            if name.isdigit():
                idx = int(name)
                while len(data) <= idx:
                    data.append(None)
                data[idx] = value
        else:
            setattr(data, name, value)

    def __getitem__(self, key):
        data = self._data
        currentpath = self._currentpath
        attr = self._attr
        schema = self._schema
        root = self._root

        if isinstance(key, str) and key.startswith("$"):
            val = jsonpath(data, key)
            newobj = jdict.__new__(jdict)
            object.__setattr__(newobj, "_data", val)
            object.__setattr__(newobj, "_attr", attr)
            object.__setattr__(newobj, "_schema", schema)
            object.__setattr__(newobj, "_currentpath", key)
            object.__setattr__(newobj, "_root", root)
            object.__setattr__(newobj, "_flags", {})
            return newobj

        if isinstance(key, str) and key.startswith("{") and key.endswith("}"):
            attrname = key[1:-1]
            return self.getattr(currentpath, attrname)

        if isinstance(data, dict):
            val = data.get(key)
        elif isinstance(data, (list, tuple)):
            val = data[key]
        else:
            val = data[key] if hasattr(data, "__getitem__") else None

        if isinstance(key, str):
            escapedkey = _esckey(key)
            newpath = f"{currentpath}.{escapedkey}"
        else:
            newpath = f"{currentpath}[{key}]"

        newobj = jdict.__new__(jdict)
        object.__setattr__(newobj, "_data", val)
        object.__setattr__(newobj, "_attr", attr)
        object.__setattr__(newobj, "_schema", schema)
        object.__setattr__(newobj, "_currentpath", newpath)
        object.__setattr__(newobj, "_root", root)
        object.__setattr__(newobj, "_flags", {})
        return newobj

    def __setitem__(self, key, value):
        data = self._data

        if isinstance(key, str) and key.startswith("{") and key.endswith("}"):
            attrname = key[1:-1]
            self.setattr(self._currentpath, attrname, value)
            return

        if isinstance(key, str) and key.startswith("$"):
            jsonpath(data, key, value)
            return

        if data is None:
            data = {}
            object.__setattr__(self, "_data", data)

        if isinstance(data, dict):
            data[key] = value
        elif isinstance(data, list):
            if isinstance(key, int):
                while len(data) <= key:
                    data.append(None)
                data[key] = value
        else:
            data[key] = value

    def __le__(self, value):
        schema = self._schema
        currentpath = self._currentpath
        root = self._root

        if schema:
            subschema = jsonschema(schema, None, getsubschema=currentpath)
            if subschema:
                valid, errs = jsonschema(value, subschema, rootschema=schema)
                if not valid:
                    raise ValueError(
                        f'Schema validation failed for "{currentpath}": {"; ".join(errs)}'
                    )

        if currentpath == "$":
            object.__setattr__(root, "_data", value)
        else:
            path = re.sub(r"^\$\.?", "", currentpath)
            if path:
                parts = _split_path(path)
                _set_nested(root._data, parts, value)
        return root

    def tojson(self, **kwargs):
        try:
            from jdata import save as jd_save

            return jd_save(self._data, **kwargs)
        except:
            return json.dumps(self._data, **kwargs)

    def fromjson(self, source, **kwargs):
        try:
            from jdata import load as jd_load

            object.__setattr__(self, "_data", jd_load(source, **kwargs))
        except:
            if isinstance(source, str):
                if source.startswith("{") or source.startswith("["):
                    object.__setattr__(self, "_data", json.loads(source))
                else:
                    with open(source, "r") as f:
                        object.__setattr__(self, "_data", json.load(f))
        return self

    def keys(self):
        data = self._data
        if isinstance(data, dict):
            return list(data.keys())
        elif isinstance(data, (list, tuple)):
            return list(range(len(data)))
        return []

    def len(self):
        data = self._data
        if isinstance(data, dict):
            return len(data)
        elif hasattr(data, "__len__"):
            return len(data)
        return 0

    def size(self):
        data = self._data
        if isinstance(data, np.ndarray):
            return data.shape
        if isinstance(data, (list, tuple)):
            if data and isinstance(data[0], (list, tuple)):
                return (len(data), len(data[0]))
            return (len(data),)
        return ()

    def iskey(self, key):
        data = self._data
        if isinstance(data, dict):
            return key in data
        elif isinstance(data, (list, tuple)):
            return isinstance(key, int) and 0 <= key < len(data)
        return False

    def isfield(self, key):
        return self.iskey(key)

    def rmfield(self, key):
        data = self._data
        if isinstance(data, dict) and key in data:
            del data[key]
        elif isinstance(data, list) and isinstance(key, int):
            del data[key]

    def setattr(self, *args):
        if len(args) == 2:
            attrname, attrvalue = args
            datapath = self._currentpath
        else:
            datapath, attrname, attrvalue = args

        attr = object.__getattribute__(self, "_attr")
        if datapath not in attr:
            attr[datapath] = {}
        attr[datapath][attrname] = attrvalue
        return attr

    def getattr(self, *args):
        attr = object.__getattribute__(self, "_attr")
        currentpath = object.__getattribute__(self, "_currentpath")

        if len(args) == 0:
            if currentpath not in attr and currentpath == "$":
                return list(attr.keys())
            elif currentpath in attr:
                return list(attr[currentpath].keys())
            return None

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str) and not arg.startswith("$"):
                attrname = arg
                datapath = currentpath
            else:
                datapath = arg
                attrname = None
        else:
            datapath, attrname = args

        if datapath not in attr:
            return None
        attrmap = attr[datapath]
        if attrname is None:
            return attrmap
        return attrmap.get(attrname)

    def setschema(self, schemadata):
        if schemadata is None:
            object.__setattr__(self, "_schema", None)
            return self

        if isinstance(schemadata, dict):
            object.__setattr__(self, "_schema", schemadata)
        elif isinstance(schemadata, str):
            try:
                from jdata import load as jd_load

                object.__setattr__(self, "_schema", jd_load(schemadata))
            except:
                if schemadata.startswith("{"):
                    object.__setattr__(self, "_schema", json.loads(schemadata))
                else:
                    with open(schemadata, "r") as f:
                        object.__setattr__(self, "_schema", json.load(f))
        return self

    def getschema(self, fmt=None):
        schema = self._schema
        if schema is None:
            return None
        if fmt and fmt.lower() in ("json", "string"):
            return json.dumps(schema)
        return schema

    def validate(self, schemadata=None):
        if schemadata is not None:
            self.setschema(schemadata)

        schema = self._schema
        if schema is None:
            raise ValueError("No schema available. Use setschema() first.")

        currentpath = self._currentpath
        subschema = jsonschema(schema, None, getsubschema=currentpath)

        if not subschema:
            return []

        valid, errors = jsonschema(self._data, subschema, rootschema=schema)
        return errors

    def attr2schema(self, **kwargs):
        schema_keywords = {
            "type",
            "enum",
            "const",
            "default",
            "binType",
            "minDims",
            "maxDims",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "contains",
            "prefixItems",
            "properties",
            "required",
            "additionalProperties",
            "minProperties",
            "maxProperties",
            "patternProperties",
            "propertyNames",
            "dependentRequired",
            "dependentSchemas",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "title",
            "description",
            "examples",
            "$comment",
            "$ref",
            "$defs",
            "definitions",
        }

        schema = {}
        basepath = self._currentpath
        attr = self._attr

        if basepath == "$":
            if "title" in kwargs:
                schema["title"] = kwargs["title"]
            if "description" in kwargs:
                schema["description"] = kwargs["description"]

        if basepath in attr:
            pathattrs = attr[basepath]
            for aname, aval in pathattrs.items():
                if aname.startswith(":"):
                    keyword = aname[1:]
                    if keyword in schema_keywords:
                        schema[keyword] = aval

        childpaths = []
        baselen = len(basepath)
        for p in attr.keys():
            if len(p) > baselen and p.startswith(basepath):
                if basepath == "$":
                    if len(p) > 2 and p[1] == ".":
                        remainder = p[2:]
                    else:
                        continue
                else:
                    if len(p) > baselen + 1 and p[baselen] == ".":
                        remainder = p[baselen + 1 :]
                    else:
                        continue
                unescaped = remainder.replace("\\.", "")
                if "." not in unescaped:
                    childpaths.append(p)

        if childpaths:
            if "type" not in schema:
                schema["type"] = "object"

            properties = {}
            for childpath in childpaths:
                if basepath == "$":
                    propname = childpath[2:]
                else:
                    propname = childpath[baselen + 1 :]
                propname = propname.replace("\\.", ".")

                tempobj = jdict.__new__(jdict)
                object.__setattr__(tempobj, "_data", None)
                object.__setattr__(tempobj, "_attr", attr)
                object.__setattr__(tempobj, "_schema", None)
                object.__setattr__(tempobj, "_currentpath", childpath)
                object.__setattr__(tempobj, "_root", tempobj)
                object.__setattr__(tempobj, "_flags", {})
                properties[propname] = tempobj.attr2schema()

            schema["properties"] = properties

        if "type" not in schema:
            if childpaths or basepath == "$":
                schema["type"] = "object"

        return schema

    def __repr__(self):
        return f"jdict({self._data!r})"

    def __str__(self):
        return str(self._data)


class _DimAccessor:
    """Helper class for dimension-based indexing like jd.data.x('label')"""

    __slots__ = ("_parent", "_dimname")

    def __init__(self, parent, dimname):
        self._parent = parent
        self._dimname = dimname

    def __call__(self, sel):
        p = self._parent
        dims = _get_attr_value(p._attr, p._currentpath, "dims")
        data = p._data
        if not isinstance(data, np.ndarray) or not dims:
            return None

        # Get current position of this dim
        dimpos = dims.index(self._dimname)

        # Build index tuple
        idx = [slice(None)] * data.ndim
        coords = _get_attr_value(p._attr, p._currentpath, "coords")
        idx[dimpos] = (
            _coordlookup(coords.get(self._dimname), sel, self._dimname)
            if coords
            else sel
        )

        # Slice and build new jdict
        result = data[tuple(idx)]
        is_scalar = isinstance(idx[dimpos], (int, np.integer))

        # Update dims/coords for cascade (remove dim if scalar selection)
        new_attr = {"$": {}}
        new_attr["$"]["dims"] = [
            d for d in dims if not (is_scalar and d == self._dimname)
        ]
        if coords:
            new_attr["$"]["coords"] = {
                k: v
                for k, v in coords.items()
                if not (is_scalar and k == self._dimname)
            }

        newobj = jdict.__new__(jdict)
        for attr, val in [
            ("_data", result),
            ("_attr", new_attr),
            ("_schema", p._schema),
            ("_currentpath", "$"),
            ("_root", None),
            ("_flags", {}),
        ]:
            object.__setattr__(newobj, attr, val)
        object.__setattr__(newobj, "_root", newobj)
        return newobj


def _get_attr_value(attr, path, name):
    if path in attr and name in attr[path]:
        return attr[path][name]
    return None


def _coordlookup(coords, sel, dimname):
    """Convert coordinate labels to indices."""
    if coords is None:
        return sel

    coords_arr = np.asarray(coords)
    is_numeric_coords = np.issubdtype(coords_arr.dtype, np.number)

    # Numeric value(s) on numeric coords -> lookup
    if is_numeric_coords and isinstance(
        sel, (int, float, np.number, list, tuple, np.ndarray)
    ):
        if isinstance(sel, (int, float, np.number)):
            idx = np.where(coords_arr == sel)[0]
            if len(idx) == 0:
                raise ValueError(f'Coord {sel} not found in "{dimname}"')
            return int(idx[0])
        elif all(isinstance(s, (int, float, np.number)) for s in sel):
            return [int(np.where(coords_arr == s)[0][0]) for s in sel]

    # Int on non-numeric coords -> direct index
    if isinstance(sel, (int, np.integer)) and not is_numeric_coords:
        return sel

    # Slice dict -> slice object
    if isinstance(sel, dict) and "start" in sel:
        coords_list = coords_arr.tolist()
        start = coords_list.index(sel["start"]) if sel.get("start") else 0
        stop = (
            coords_list.index(sel["stop"]) + 1 if sel.get("stop") else len(coords_list)
        )
        return slice(start, stop)

    # String or list of strings -> index lookup
    coords_list = coords_arr.tolist()
    if isinstance(sel, (list, tuple)):
        return [coords_list.index(s) for s in sel]
    return coords_list.index(sel)


def _esckey(key):
    """Escape dots in key for JSONPath - Python compatible version."""
    if "." not in key:
        return key
    result = []
    for i, ch in enumerate(key):
        if ch == "." and (i == 0 or key[i - 1] != "\\"):
            result.append("\\.")
        else:
            result.append(ch)
    return "".join(result)


def _split_path(path):
    parts = []
    current = ""
    i = 0
    while i < len(path):
        if i < len(path) - 1 and path[i] == "\\" and path[i + 1] == ".":
            current += "."
            i += 2
        elif path[i] == ".":
            if current:
                parts.append(current)
            current = ""
            i += 1
        elif path[i] == "[":
            if current:
                parts.append(current)
            current = ""
            end = path.find("]", i)
            if end > i:
                parts.append(int(path[i + 1 : end]))
                i = end + 1
            else:
                i += 1
        else:
            current += path[i]
            i += 1
    if current:
        parts.append(current)
    return parts


def _set_nested(data, parts, value):
    for i, part in enumerate(parts[:-1]):
        if isinstance(part, int):
            while len(data) <= part:
                data.append({})
            if data[part] is None:
                data[part] = {}
            data = data[part]
        else:
            if part not in data or data[part] is None:
                data[part] = {}
            data = data[part]

    lastpart = parts[-1]
    if isinstance(lastpart, int):
        while len(data) <= lastpart:
            data.append(None)
        data[lastpart] = value
    else:
        data[lastpart] = value


def _simple_jsonpath(data, path):
    if path == "$":
        return data

    path = re.sub(r"^\$\.?", "", path)
    if not path:
        return data

    if ".." in path:
        idx = path.find("..")
        before = path[:idx]
        after = path[idx + 2 :]

        if before:
            current = _simple_jsonpath(data, "$." + before)
        else:
            current = data

        if after:
            key = after.split(".")[0].split("[")[0]
            results = _deep_scan(current, key)
            rest = after[len(key) :]
            if rest:
                final = []
                for r in results:
                    if rest.startswith("."):
                        final.append(_simple_jsonpath(r, "$" + rest))
                    else:
                        final.append(r)
                return final
            return results
        return current

    parts = _split_path(path)
    current = data
    for part in parts:
        if current is None:
            return None
        if isinstance(part, int):
            if isinstance(current, (list, tuple)) and part < len(current):
                current = current[part]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)

    return current


def _simple_jsonpath_set(data, path, value):
    path = re.sub(r"^\$\.?", "", path)
    if not path:
        return
    parts = _split_path(path)
    _set_nested(data, parts, value)


def _deep_scan(data, key):
    results = []

    def scan(obj):
        if isinstance(obj, dict):
            if key in obj:
                results.append(obj[key])
            for v in obj.values():
                scan(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                scan(item)

    scan(data)
    return results


def _getkindschema(kind):
    def intschema(mn, mx):
        return {"type": "integer", "minimum": mn, "maximum": mx}

    def objschema(props, req):
        return {"type": "object", "properties": props, "required": req}

    schemas = {
        "uuid": objschema(
            {
                "time_low": intschema(0, 4294967295),
                "time_mid": intschema(0, 65535),
                "time_high": intschema(0, 65535),
                "clock_seq": intschema(0, 65535),
                "node": intschema(0, 281474976710655),
            },
            ["time_low", "time_mid", "time_high", "clock_seq", "node"],
        ),
        "date": objschema(
            {
                "year": intschema(1, 9999),
                "month": intschema(1, 12),
                "day": intschema(1, 31),
            },
            ["year", "month", "day"],
        ),
        "time": objschema(
            {
                "hour": intschema(0, 23),
                "min": intschema(0, 59),
                "sec": {"type": "number", "minimum": 0, "exclusiveMaximum": 60},
            },
            ["hour", "min", "sec"],
        ),
        "datetime": objschema(
            {
                "year": intschema(1, 9999),
                "month": intschema(1, 12),
                "day": intschema(1, 31),
                "hour": intschema(0, 23),
                "min": intschema(0, 59),
                "sec": {"type": "number", "minimum": 0, "exclusiveMaximum": 60},
            },
            ["year", "month", "day", "hour", "min", "sec"],
        ),
        "email": objschema(
            {
                "user": {"type": "string", "minLength": 1},
                "domain": {"type": "string", "pattern": r"^[^@\s]+\.[^@\s]+$"},
            },
            ["user", "domain"],
        ),
        "uri": objschema(
            {
                "scheme": {"type": "string", "pattern": r"^[a-zA-Z][a-zA-Z0-9+.-]*$"},
                "host": {"type": "string", "minLength": 1},
                "port": intschema(0, 65535),
                "path": {"type": "string"},
                "query": {"type": "string"},
                "fragment": {"type": "string"},
            },
            ["scheme", "host"],
        ),
    }

    bintypes = [
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
        "bool",
    ]

    if kind.lower() in schemas:
        return schemas[kind.lower()]
    elif kind.lower() in bintypes:
        return {"binType": kind.lower()}
    return None


def _formatkind(kind, data):
    if not isinstance(data, dict):
        return None

    try:
        kind = kind.lower()
        if kind == "uuid":
            return f"{data['time_low']:08x}-{data['time_mid']:04x}-{data['time_high']:04x}-{data['clock_seq']:04x}-{data['node']:012x}"
        elif kind == "date":
            return f"{data['year']:04d}-{data['month']:02d}-{data['day']:02d}"
        elif kind == "time":
            sec = data["sec"]
            if sec == int(sec):
                return f"{data['hour']:02d}:{data['min']:02d}:{int(sec):02d}"
            return f"{data['hour']:02d}:{data['min']:02d}:{sec:06.3f}"
        elif kind == "datetime":
            sec = data["sec"]
            if sec == int(sec):
                return f"{data['year']:04d}-{data['month']:02d}-{data['day']:02d}T{data['hour']:02d}:{data['min']:02d}:{int(sec):02d}"
            return f"{data['year']:04d}-{data['month']:02d}-{data['day']:02d}T{data['hour']:02d}:{data['min']:02d}:{sec:06.3f}"
        elif kind == "email":
            return f"{data['user']}@{data['domain']}"
        elif kind == "uri":
            s = f"{data['scheme']}://{data['host']}"
            if "port" in data and data["port"]:
                s += f":{data['port']}"
            if "path" in data:
                s += data["path"]
            if "query" in data and data["query"]:
                s += f"?{data['query']}"
            if "fragment" in data and data["fragment"]:
                s += f"#{data['fragment']}"
            return s
    except (KeyError, TypeError):
        pass
    return None
