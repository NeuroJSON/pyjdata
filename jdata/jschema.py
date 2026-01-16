"""
jsonschema - Validate Python data structures against JSON Schema (draft-07 compatible)

Author: Qianqian Fang (q.fang at neu.edu)
Converted from MATLAB by Claude

Example:
    data = {'name': 'John', 'age': 30}
    schema = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}}
    valid, errors = jsonschema(data, schema)
"""

__all__ = [
    "jsonschema",
]

##====================================================================================
## dependent libraries
##====================================================================================

import json
import re
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

_BINTYPES = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "uint64": np.uint64,
    "int64": np.int64,
    "float32": np.float32,
    "single": np.float32,
    "float64": np.float64,
    "double": np.float64,
    "bool": np.bool_,
    "logical": np.bool_,
}


def jsonschema(
    data: Any, schema: Any = None, **kwargs
) -> Union[Tuple[bool, List[str]], Any]:
    """
    Validate data against JSON Schema or generate data from schema.

    Args:
        data: Data to validate, or schema if generating
        schema: JSON Schema (dict, JSON string, file path, or URL)
        **kwargs:
            rootschema: Root schema for resolving $ref
            generate: Generation mode - 'all', 'required', 'requireddefaults'
            getsubschema: Get subschema at JSONPath

    Returns:
        For validation: (valid, errors) tuple
        For generation: generated data (not a tuple)
        For getsubschema: the subschema (not a tuple)
    """
    if "resolveref" in kwargs:
        return _resolveref(kwargs["resolveref"], data)

    if "getsubschema" in kwargs:
        return _getsubschema(data, kwargs["getsubschema"])

    # Generation mode: jsonschema(schema) or jsonschema(schema, generate='...')
    if schema is None or "generate" in kwargs:
        schemaarg = data if schema is None else data
        if isinstance(schemaarg, str):
            schemaarg = _load_schema(schemaarg)
        opts = {"rootschema": schemaarg, **kwargs}
        if "generate" not in opts:
            opts["generate"] = "requireddefaults"
        return _generatedata(schemaarg, opts)

    # Validation mode
    if isinstance(schema, str):
        schema = _load_schema(schema)

    opts = {"rootschema": kwargs.get("rootschema", schema), **kwargs}
    return _validatedata(data, schema, "$", opts)


def _load_schema(schema_source: str) -> dict:
    if schema_source.startswith("{") or schema_source.startswith("["):
        return json.loads(schema_source)
    elif schema_source.startswith("http://") or schema_source.startswith("https://"):
        from urllib.request import urlopen

        with urlopen(schema_source) as resp:
            return json.loads(resp.read().decode())
    else:
        with open(schema_source, "r") as f:
            return json.load(f)


def _validatedata(
    data: Any, schema: Any, path: str, opts: dict
) -> Tuple[bool, List[str]]:
    rootschema = opts.get("rootschema", schema)
    valid = True
    errors = []

    if isinstance(schema, bool):
        if not schema:
            return False, [f"{path}: schema is false"]
        return True, []

    if not isinstance(schema, dict) or not schema:
        return True, []

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#"):
            refschema = _resolveref(ref, rootschema)
            if refschema is not None:
                return _validatedata(data, refschema, path, opts)
            return False, [f'{path}: cannot resolve $ref "{ref}"']
        return True, []

    # type validation
    if "type" in schema:
        schematype = schema["type"]
        if isinstance(schematype, list):
            if not any(_checktype(data, t) for t in schematype):
                valid = False
                errors.append(f"{path}: type mismatch")
        elif not _checktype(data, schematype):
            valid = False
            errors.append(f"{path}: expected {schematype}, got {_gettype(data)}")

    # enum validation
    if "enum" in schema:
        enumvals = schema["enum"]
        match = False
        for ev in enumvals:
            if _values_equal(data, ev):
                match = True
                break
        if not match:
            valid = False
            errors.append(f"{path}: not in enum")

    # const validation
    if "const" in schema:
        if not _values_equal(data, schema["const"]):
            valid = False
            errors.append(f"{path}: const mismatch")

    # numeric validation
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        isvalid, errmsg = _validatenumeric(data, schema, path)
        if not isvalid:
            valid = False
            errors.extend(errmsg)

    # numpy array validation
    if isinstance(data, np.ndarray) or "binType" in schema:
        isvalid, errmsg = _validatebinary(data, schema, path)
        if not isvalid:
            valid = False
            errors.extend(errmsg)

    # string validation
    if isinstance(data, str):
        isvalid, errmsg = _validatestring(data, schema, path)
        if not isvalid:
            valid = False
            errors.extend(errmsg)

    # array validation
    if _isarray(data):
        isvalid, errmsg = _validatearray(data, schema, path, opts)
        if not isvalid:
            valid = False
            errors.extend(errmsg)

    # object validation
    if isinstance(data, dict):
        isvalid, errmsg = _validateobject(data, schema, path, opts)
        if not isvalid:
            valid = False
            errors.extend(errmsg)

    # composition validation
    isvalid, errmsg = _validatecomposition(data, schema, path, opts)
    if not isvalid:
        valid = False
        errors.extend(errmsg)

    # if/then/else
    if "if" in schema:
        ifok, _ = _validatedata(data, schema["if"], path, opts)
        subkey = None
        if ifok and "then" in schema:
            subkey = "then"
        elif not ifok and "else" in schema:
            subkey = "else"
        if subkey:
            isvalid, errmsg = _validatedata(data, schema[subkey], path, opts)
            if not isvalid:
                valid = False
                errors.extend(errmsg)

    return valid, errors


def _values_equal(a, b):
    """Compare two values for equality, handling empty strings and None."""
    if a is None and b is None:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    # Handle empty string vs None
    a_empty = (a is None) or (isinstance(a, str) and a == "")
    b_empty = (b is None) or (isinstance(b, str) and b == "")
    if a_empty and b_empty:
        return True
    try:
        return a == b
    except:
        return False


def _checktype(data: Any, schematype: str) -> bool:
    if schematype == "null":
        return data is None
    elif schematype == "boolean":
        return isinstance(data, bool)
    elif schematype == "integer":
        if isinstance(data, bool):
            return False
        if isinstance(data, int):
            return True
        if isinstance(data, float):
            return data == int(data)
        return False
    elif schematype == "number":
        if isinstance(data, bool):
            return False
        return isinstance(data, (int, float))
    elif schematype == "string":
        return isinstance(data, str)
    elif schematype == "array":
        if isinstance(data, (list, tuple)):
            return True
        if isinstance(data, np.ndarray):
            return True
        return False
    elif schematype == "object":
        return isinstance(data, dict)
    return True


def _gettype(data: Any) -> str:
    if data is None:
        return "null"
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, int):
        return "integer"
    if isinstance(data, float):
        return "number"
    if isinstance(data, str):
        return "string"
    if isinstance(data, (list, tuple)):
        return "array"
    if isinstance(data, dict):
        return "object"
    if isinstance(data, np.ndarray):
        return "array"
    return "unknown"


def _isarray(data: Any) -> bool:
    if isinstance(data, (list, tuple)):
        return True
    if isinstance(data, np.ndarray) and data.ndim >= 1:
        return True
    return False


def _resolveref(ref: str, root: dict) -> Optional[dict]:
    ptr = ref[1:] if ref.startswith("#") else ref
    if not ptr:
        return root
    if ptr.startswith("/"):
        ptr = ptr[1:]

    parts = ptr.split("/")
    current = root

    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            except ValueError:
                return None
        else:
            return None

    return current


def _validatenumeric(
    data: Union[int, float], schema: dict, path: str
) -> Tuple[bool, List[str]]:
    valid = True
    errors = []

    if "minimum" in schema and data < schema["minimum"]:
        valid = False
        errors.append(f"{path}: value < minimum")
    if "maximum" in schema and data > schema["maximum"]:
        valid = False
        errors.append(f"{path}: value > maximum")
    if "exclusiveMinimum" in schema and data <= schema["exclusiveMinimum"]:
        valid = False
        errors.append(f"{path}: value <= exclusiveMinimum")
    if "exclusiveMaximum" in schema and data >= schema["exclusiveMaximum"]:
        valid = False
        errors.append(f"{path}: value >= exclusiveMaximum")
    if "multipleOf" in schema:
        mult = schema["multipleOf"]
        if mult > 0:
            remainder = abs(data % mult)
            eps = 1e-10 * max(abs(data), 1)
            if remainder > eps and abs(remainder - mult) > eps:
                valid = False
                errors.append(f"{path}: not multipleOf {mult}")

    return valid, errors


def _validatebinary(data, schema: dict, path: str) -> Tuple[bool, List[str]]:
    """Validate binary/array data against binType and dims."""
    valid, errors = True, []

    if "binType" in schema:
        dtype = _BINTYPES.get(schema["binType"])
        if dtype is None:
            return False, [f'{path}: invalid binType "{schema["binType"]}"']
        if not isinstance(data, np.ndarray):
            return False, [f"{path}: expected numpy array, got {type(data).__name__}"]
        if data.dtype != dtype:
            return False, [f"{path}: expected {schema['binType']}, got {data.dtype}"]

    if not isinstance(data, np.ndarray):
        return valid, errors

    # Validate minDims/maxDims
    for dimtype in ("minDims", "maxDims"):
        if dimtype not in schema:
            continue
        dims = schema[dimtype]
        dims = [int(dims)] if isinstance(dims, (int, float)) else [int(d) for d in dims]
        ismin = dimtype == "minDims"

        if len(dims) == 1:  # Vector check
            actual = (
                max(data.shape)
                if data.ndim <= 2 and (data.ndim == 1 or 1 in data.shape)
                else -1
            )
            if actual < 0:
                valid, errors = False, errors + [f"{path}: expected 1D array"]
            elif (ismin and actual < dims[0]) or (not ismin and actual > dims[0]):
                valid, errors = False, errors + [
                    f"{path}: length {actual} violates {dimtype} {dims[0]}"
                ]
        else:  # ND check
            for i, d in enumerate(dims):
                actual = data.shape[i] if i < data.ndim else 1
                if (ismin and actual < d) or (not ismin and actual > d):
                    valid, errors = False, errors + [
                        f"{path}: dim {i} is {actual}, violates {dimtype} {d}"
                    ]

    return valid, errors


def _validatestring(data: str, schema: dict, path: str) -> Tuple[bool, List[str]]:
    valid = True
    errors = []

    if "minLength" in schema and len(data) < schema["minLength"]:
        valid = False
        errors.append(f"{path}: string too short")
    if "maxLength" in schema and len(data) > schema["maxLength"]:
        valid = False
        errors.append(f"{path}: string too long")
    if "pattern" in schema and not re.search(schema["pattern"], data):
        valid = False
        errors.append(f"{path}: pattern mismatch")
    if "format" in schema:
        fmt = schema["format"]
        patterns = {
            "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            "uri": r"^https?://",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "ipv4": r"^(\d{1,3}\.){3}\d{1,3}$",
            "uuid": r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$",
        }
        if fmt in patterns and not re.search(patterns[fmt], data):
            valid = False
            errors.append(f"{path}: invalid {fmt}")

    return valid, errors


def _validatearray(data, schema: dict, path: str, opts: dict) -> Tuple[bool, List[str]]:
    valid = True
    errors = []

    # Get array length and element accessor
    if isinstance(data, np.ndarray):
        length = data.shape[0] if data.ndim > 0 else 0
        getelem = lambda i: data[i]
    else:
        length = len(data)
        getelem = lambda i: data[i]

    if "minItems" in schema and length < schema["minItems"]:
        valid = False
        errors.append(f"{path}: too few items")
    if "maxItems" in schema and length > schema["maxItems"]:
        valid = False
        errors.append(f"{path}: too many items")
    if schema.get("uniqueItems"):
        for i in range(length):
            for j in range(i + 1, length):
                try:
                    if _values_equal(getelem(i), getelem(j)):
                        valid = False
                        errors.append(f"{path}: duplicate items")
                        break
                except:
                    pass
            if not valid:
                break

    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            for i in range(min(length, len(items))):
                isvalid, errmsg = _validatedata(
                    getelem(i), items[i], f"{path}[{i}]", opts
                )
                if not isvalid:
                    valid = False
                    errors.extend(errmsg)
        else:
            for i in range(length):
                isvalid, errmsg = _validatedata(getelem(i), items, f"{path}[{i}]", opts)
                if not isvalid:
                    valid = False
                    errors.extend(errmsg)

    if "contains" in schema:
        found = False
        for i in range(length):
            isvalid, _ = _validatedata(getelem(i), schema["contains"], path, opts)
            if isvalid:
                found = True
                break
        if not found:
            valid = False
            errors.append(f"{path}: contains not satisfied")

    return valid, errors


def _validateobject(
    data: dict, schema: dict, path: str, opts: dict
) -> Tuple[bool, List[str]]:
    valid = True
    errors = []
    numkeys = len(data)

    if "minProperties" in schema and numkeys < schema["minProperties"]:
        valid = False
        errors.append(f"{path}: too few properties")
    if "maxProperties" in schema and numkeys > schema["maxProperties"]:
        valid = False
        errors.append(f"{path}: too many properties")
    if "required" in schema:
        for req in schema["required"]:
            if req not in data:
                valid = False
                errors.append(f'{path}: missing "{req}"')

    validatedkeys = set()

    if "properties" in schema:
        props = schema["properties"]
        for pname, pschema in props.items():
            if pname in data:
                validatedkeys.add(pname)
                isvalid, errmsg = _validatedata(
                    data[pname], pschema, f"{path}.{pname}", opts
                )
                if not isvalid:
                    valid = False
                    errors.extend(errmsg)

    if "patternProperties" in schema:
        patternprops = schema["patternProperties"]
        for keyname in data.keys():
            for pattern, pschema in patternprops.items():
                if re.search(pattern, keyname):
                    validatedkeys.add(keyname)
                    isvalid, errmsg = _validatedata(
                        data[keyname], pschema, f"{path}.{keyname}", opts
                    )
                    if not isvalid:
                        valid = False
                        errors.extend(errmsg)

    if "additionalProperties" in schema:
        addprops = schema["additionalProperties"]
        for keyname in data.keys():
            if keyname not in validatedkeys:
                if addprops is False:
                    valid = False
                    errors.append(f'{path}: extra property "{keyname}"')
                elif isinstance(addprops, dict):
                    isvalid, errmsg = _validatedata(
                        data[keyname], addprops, f"{path}.{keyname}", opts
                    )
                    if not isvalid:
                        valid = False
                        errors.extend(errmsg)

    return valid, errors


def _validatecomposition(
    data: Any, schema: dict, path: str, opts: dict
) -> Tuple[bool, List[str]]:
    valid = True
    errors = []

    if "allOf" in schema:
        for subschema in schema["allOf"]:
            isvalid, errmsg = _validatedata(data, subschema, path, opts)
            if not isvalid:
                valid = False
                errors.extend(errmsg)

    if "anyOf" in schema:
        match = False
        for subschema in schema["anyOf"]:
            isvalid, _ = _validatedata(data, subschema, path, opts)
            if isvalid:
                match = True
                break
        if not match:
            valid = False
            errors.append(f"{path}: anyOf not satisfied")

    if "oneOf" in schema:
        matchcount = sum(
            1 for s in schema["oneOf"] if _validatedata(data, s, path, opts)[0]
        )
        if matchcount != 1:
            valid = False
            errors.append(f"{path}: oneOf matched {matchcount}")

    if "not" in schema:
        isvalid, _ = _validatedata(data, schema["not"], path, opts)
        if isvalid:
            valid = False
            errors.append(f"{path}: not violated")

    return valid, errors


def _generatedata(schema: dict, opts: dict) -> Any:
    rootschema = opts.get("rootschema", schema)
    genopt = opts.get("generate", "requireddefaults")

    if not isinstance(schema, dict) or not schema:
        return None

    if "$ref" in schema:
        refschema = _resolveref(schema["$ref"], rootschema)
        if refschema:
            return _generatedata(refschema, opts)
        return None

    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    schematype = schema.get("type", "")
    if isinstance(schematype, list):
        schematype = schematype[0] if schematype else ""
    if not schematype:
        if "properties" in schema or "required" in schema:
            schematype = "object"
        elif "items" in schema:
            schematype = "array"

    if "binType" in schema:
        dtype = _BINTYPES.get(schema["binType"], np.float64)
        dims = schema.get("minDims", 1)
        dims = (
            (int(dims),)
            if isinstance(dims, (int, float))
            else tuple(int(d) for d in dims)
        )
        return np.zeros(dims, dtype=dtype)

    if schematype == "null":
        return None
    elif schematype == "boolean":
        return False
    elif schematype == "integer":
        return _generateinteger(schema)
    elif schematype == "number":
        return _generatenumber(schema)
    elif schematype == "string":
        return _generatestring(schema)
    elif schematype == "array":
        return _generatearray(schema, opts)
    elif schematype == "object":
        return _generateobject(schema, opts)

    if "allOf" in schema:
        result = {}
        for subschema in schema["allOf"]:
            subdata = _generatedata(subschema, opts)
            if isinstance(result, dict) and isinstance(subdata, dict):
                result.update(subdata)
        return result

    return None


def _generateinteger(schema: dict) -> int:
    val = 0
    if "minimum" in schema:
        val = schema["minimum"]
    if "exclusiveMinimum" in schema:
        excmin = schema["exclusiveMinimum"]
        if val <= excmin:
            val = int(excmin) + 1
    val = int(math.ceil(val))
    if "multipleOf" in schema:
        mult = schema["multipleOf"]
        if mult > 0:
            val = int(math.ceil(val / mult)) * mult
    return val


def _generatenumber(schema: dict) -> float:
    val = 0.0
    if "minimum" in schema:
        val = float(schema["minimum"])
    if "exclusiveMinimum" in schema:
        excmin = schema["exclusiveMinimum"]
        if val <= excmin:
            val = excmin + 1e-10
    if "multipleOf" in schema:
        mult = schema["multipleOf"]
        if mult > 0:
            val = math.ceil(val / mult) * mult
    return val


def _generatestring(schema: dict) -> str:
    val = ""
    if "format" in schema:
        formats = {
            "email": "user@example.com",
            "uri": "http://example.com",
            "date": "2000-01-01",
            "ipv4": "0.0.0.0",
            "uuid": "00000000-0000-0000-0000-000000000000",
        }
        val = formats.get(schema["format"], "")
    if "minLength" in schema:
        minlen = schema["minLength"]
        if len(val) < minlen:
            val += "a" * (minlen - len(val))
    return val


def _generatearray(schema: dict, opts: dict) -> list:
    val = []
    minitems = schema.get("minItems", 0)

    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            for itemschema in items:
                val.append(_generatedata(itemschema, opts))
        else:
            for _ in range(minitems):
                val.append(_generatedata(items, opts))
    else:
        val = [None] * minitems

    return val


def _generateobject(schema: dict, opts: dict) -> dict:
    genopt = opts.get("generate", "requireddefaults")
    val = {}
    reqfields = schema.get("required", [])

    if "properties" in schema:
        props = schema["properties"]
        for pname, pschema in props.items():
            isreq = pname in reqfields
            hasdefault = isinstance(pschema, dict) and "default" in pschema

            shouldgen = False
            if genopt == "all":
                shouldgen = True
            elif genopt == "required":
                shouldgen = isreq
            elif genopt == "requireddefaults":
                shouldgen = isreq or hasdefault

            if shouldgen:
                if isinstance(pschema, dict):
                    val[pname] = _generatedata(pschema, opts)
                else:
                    val[pname] = None
    else:
        for req in reqfields:
            val[req] = None

    return val


def _getsubschema(schema: dict, jsonpath: str) -> Optional[dict]:
    if not schema or not jsonpath or jsonpath == "$":
        return schema

    path = re.sub(r"^\$\.?", "", jsonpath)
    if not path:
        return schema

    subschema = schema

    # Simple tokenizer for path
    tokens = []
    i = 0
    while i < len(path):
        if path[i] == "[":
            end = path.find("]", i)
            if end > i:
                tokens.append(path[i : end + 1])
                i = end + 1
            else:
                i += 1
        elif path[i] == ".":
            i += 1
        elif i < len(path) - 1 and path[i] == "\\" and path[i + 1] == ".":
            # Escaped dot - start of key with dot
            current = ""
            while i < len(path):
                if i < len(path) - 1 and path[i] == "\\" and path[i + 1] == ".":
                    current += "."
                    i += 2
                elif path[i] == "." or path[i] == "[":
                    break
                else:
                    current += path[i]
                    i += 1
            if current:
                tokens.append(current)
        else:
            # Regular key
            current = ""
            while i < len(path) and path[i] != "." and path[i] != "[":
                if i < len(path) - 1 and path[i] == "\\" and path[i + 1] == ".":
                    current += "."
                    i += 2
                else:
                    current += path[i]
                    i += 1
            if current:
                tokens.append(current)

    for tok in tokens:
        # Resolve $ref if present
        while isinstance(subschema, dict) and "$ref" in subschema:
            subschema = _resolveref(subschema["$ref"], schema)
            if subschema is None:
                return None

        if tok.startswith("["):
            # Array index -> use items schema
            if isinstance(subschema, dict) and "items" in subschema:
                items = subschema["items"]
                if isinstance(items, list) and items:
                    subschema = items[0]
                else:
                    subschema = items
            else:
                return None
        else:
            # Property name
            prop = tok
            if isinstance(subschema, dict) and "properties" in subschema:
                props = subschema["properties"]
                if isinstance(props, dict) and prop in props:
                    subschema = props[prop]
                else:
                    return None
            else:
                return None

    return subschema


def coerce(data: Any, schema: dict) -> Any:
    """Coerce data to match schema's binType. For use before assignment."""
    if not isinstance(schema, dict) or "binType" not in schema:
        return data
    dtype = _BINTYPES.get(schema["binType"])
    if dtype is None or (isinstance(data, np.ndarray) and data.dtype == dtype):
        return data
    try:
        return np.asarray(data, dtype=dtype)
    except (ValueError, TypeError):
        return data
