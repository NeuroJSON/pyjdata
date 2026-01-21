"""@package docstring
File IO to load/decode GIFTI or JSON-based JGIFTI files

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "JGifti",
    "gii2jgii",
    "jgii2gii",
    "loadgifti",
    "loadjgifti",
    "savegifti",
    "savejgifti",
    "jgifticreate",
    "giicodemap",
    "get_node",
    "get_face",
    "get_property",
    "get_properties",
    "get_labels",
    "get_metadata",
    "get_coord_system",
    "get_surfaces",
]

import re
import zlib
import base64
import xml.etree.ElementTree as ET
from typing import Union, Dict, Optional
import numpy as np
from .jdata import decode as jdatadecode
from .jpath import jsonpath

# Unified lookup tables
_DATATYPE_MAP = {
    "NIFTI_TYPE_UINT8": "uint8",
    "NIFTI_TYPE_INT8": "int8",
    "NIFTI_TYPE_INT16": "int16",
    "NIFTI_TYPE_INT32": "int32",
    "NIFTI_TYPE_INT64": "int64",
    "NIFTI_TYPE_UINT16": "uint16",
    "NIFTI_TYPE_UINT32": "uint32",
    "NIFTI_TYPE_UINT64": "uint64",
    "NIFTI_TYPE_FLOAT32": "float32",
    "NIFTI_TYPE_FLOAT64": "float64",
}
_DATATYPE_MAP_REV = {v: k for k, v in _DATATYPE_MAP.items()}

_XFORM_MAP = {
    "NIFTI_XFORM_UNKNOWN": "unknown",
    "NIFTI_XFORM_SCANNER_ANAT": "scanner_anat",
    "NIFTI_XFORM_ALIGNED_ANAT": "aligned_anat",
    "NIFTI_XFORM_TALAIRACH": "talairach",
    "NIFTI_XFORM_MNI_152": "mni_152",
    "NIFTI_XFORM_TEMPLATE_OTHER": "template_other",
}
_XFORM_MAP_REV = {v: k for k, v in _XFORM_MAP.items()}

_INTENT_PROPERTY = {
    "NIFTI_INTENT_SHAPE": "Shape",
    "NIFTI_INTENT_LABEL": "Label",
    "NIFTI_INTENT_TIME_SERIES": "TimeSeries",
    "NIFTI_INTENT_RGB_VECTOR": "Color",
    "NIFTI_INTENT_RGBA_VECTOR": "Color",
    "NIFTI_INTENT_VECTOR": "Vector",
    "NIFTI_INTENT_GENMATRIX": "Tensor",
    "NIFTI_INTENT_NODE_INDEX": "NodeIndex",
    "NIFTI_INTENT_NONE": "Data",
}
for _intent in ("TTEST", "FTEST", "ZSCORE", "CORREL", "CHISQ", "BETA", "PVAL"):
    _INTENT_PROPERTY[f"NIFTI_INTENT_{_intent}"] = "Functional"

_PROPERTY_INTENT = {
    "shape": "NIFTI_INTENT_SHAPE",
    "thickness": "NIFTI_INTENT_SHAPE",
    "curvature": "NIFTI_INTENT_SHAPE",
    "sulcaldepth": "NIFTI_INTENT_SHAPE",
    "label": "NIFTI_INTENT_LABEL",
    "timeseries": "NIFTI_INTENT_TIME_SERIES",
    "functional": "NIFTI_INTENT_TTEST",
    "vector": "NIFTI_INTENT_VECTOR",
}

_IDENTITY_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def giicodemap(name: str, value: Union[str, int]) -> Union[str, int]:
    """Convert between GIFTI codes and human-readable values."""
    lut = {"xform": _XFORM_MAP, "datatype": _DATATYPE_MAP, "intent": _INTENT_PROPERTY}
    table = lut.get(name.lower())
    if not table:
        return value
    if value in table:
        return table[value]
    rev = {v: k for k, v in table.items()}
    return rev.get(value, value)


def _parse_metadata(elem: ET.Element) -> Dict[str, str]:
    """Parse MetaData element."""
    if elem is None:
        return {}
    result = {}
    for md in elem.findall("MD"):
        name_el, val_el = md.find("Name"), md.find("Value")
        if name_el is not None and name_el.text:
            result[name_el.text.strip()] = (
                (val_el.text or "").strip() if val_el is not None else ""
            )
    return result


def _parse_label_table(elem: ET.Element) -> Dict[str, Dict]:
    """Parse LabelTable element."""
    if elem is None:
        return {}
    result = {}
    for label in elem.findall("Label"):
        key = label.get("Key") or label.get("Index")
        if key:
            result[key] = {
                "Label": (label.text or "???").strip(),
                "RGBA": [
                    float(label.get(c, d))
                    for c, d in [
                        ("Red", 0.667),
                        ("Green", 0.667),
                        ("Blue", 0.667),
                        ("Alpha", 1.0),
                    ]
                ],
            }
    return result


def _parse_coord_system(elem: ET.Element) -> Dict:
    """Parse CoordinateSystemTransformMatrix element."""
    if elem is None:
        return {}
    result = {}
    for tag, key in [
        ("DataSpace", "DataSpace"),
        ("TransformedSpace", "TransformedSpace"),
    ]:
        el = elem.find(tag)
        if el is not None and el.text:
            val = el.text.strip()
            result[key] = _XFORM_MAP.get(val, val)

    mat_el = elem.find("MatrixData")
    if mat_el is not None and mat_el.text:
        vals = np.fromstring(mat_el.text, sep=" ", dtype=np.float64)
        if vals.size == 16:
            result["MatrixData"] = vals.reshape(4, 4).tolist()
    return result


def _decode_gifti_data(elem: ET.Element, attribs: Dict) -> Union[np.ndarray, Dict]:
    """Decode GIFTI Data element (from XML)."""
    encoding = attribs.get("Encoding", "ASCII")
    dtype_str = _DATATYPE_MAP.get(
        attribs.get("DataType", "NIFTI_TYPE_FLOAT32"), "float32"
    )
    endian = "<" if attribs.get("Endian", "LittleEndian") == "LittleEndian" else ">"
    dtype = np.dtype(dtype_str).newbyteorder(endian)

    dims = []
    for i in range(6):
        d = attribs.get(f"Dim{i}")
        if d is None:
            break
        dims.append(int(d))

    text = (elem.text or "").strip() if elem is not None else ""

    if encoding == "ASCII":
        arr = np.fromstring(text, sep=" ", dtype=dtype)
    elif encoding == "Base64Binary":
        arr = np.frombuffer(base64.b64decode(text), dtype=dtype)
    elif encoding == "GZipBase64Binary":
        arr = np.frombuffer(
            zlib.decompress(base64.b64decode(text), zlib.MAX_WBITS | 32), dtype=dtype
        )
    elif encoding == "ExternalFileBinary":
        return {"_DataLink_": f"file://./{attribs.get('ExternalFileName', '')}"}
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if len(dims) > 1:
        order = "F" if attribs.get("ArrayIndexingOrder") == "ColumnMajorOrder" else "C"
        arr = arr.reshape(dims, order=order)
    return arr


def _decode_jdata(data, opt: Dict = None, root: Dict = None):
    """
    Decode data that may be in JData annotated format.

    Handles:
      - Direct numpy arrays (returned as-is)
      - JData annotated dicts with _ArrayType_, _ArraySize_, _ArrayZipData_, etc.
      - _DataLink_ references with JSONPath (resolved against root)
      - _DataLink_ references to external files (via jdatadecode)
      - Regular lists/values (converted to numpy array)
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, dict):
        if "_DataLink_" in data:
            link = data["_DataLink_"]
            if isinstance(link, str) and link.startswith("$") and root is not None:
                try:
                    resolved = jsonpath(root, link)
                    return _decode_jdata(resolved, opt, root)
                except Exception:
                    pass
            decode_opt = dict(opt) if opt else {}
            decode_opt.setdefault("maxlinklevel", 1)
            return jdatadecode(data, **decode_opt)

        if "_ArrayType_" in data:
            return jdatadecode(data, **(opt or {}))
        return data

    return np.asarray(data)


def _decode_properties(props: Dict, opt: Dict = None, root: Dict = None) -> Dict:
    """Decode all properties in a Properties dict."""
    if not props:
        return props
    return {name: _decode_jdata(data, opt, root) for name, data in props.items()}


def _get_mesh_container(jgii: Dict, anatomy: str = None) -> Optional[Dict]:
    """
    Get the container holding MeshVertex3/MeshTri3.
    Handles both direct and nested (anatomy) structures.
    """
    gifti_data = jgii.get("GIFTIData")
    if not gifti_data:
        return None

    if "MeshVertex3" in gifti_data or "MeshTri3" in gifti_data:
        return gifti_data

    if anatomy:
        return gifti_data.get(anatomy)

    for key, val in gifti_data.items():
        if isinstance(val, dict) and ("MeshVertex3" in val or "MeshTri3" in val):
            return val
    return None


def get_surfaces(jgii: Dict) -> list:
    """
    Get list of surface names (anatomy identifiers).

    Returns empty list for single-surface (direct) structure.
    """
    gifti_data = jgii.get("GIFTIData", {})
    if "MeshVertex3" in gifti_data or "MeshTri3" in gifti_data:
        return []
    return [
        k
        for k, v in gifti_data.items()
        if isinstance(v, dict) and ("MeshVertex3" in v or "MeshTri3" in v)
    ]


def get_node(jgii: Dict, anatomy: str = None, opt: Dict = None) -> Optional[np.ndarray]:
    """Get vertex/node coordinates from JGIFTI structure."""
    container = _get_mesh_container(jgii, anatomy)
    if not container:
        return None
    try:
        return _decode_jdata(container["MeshVertex3"]["Data"], opt, root=jgii)
    except KeyError:
        return None


def get_face(
    jgii: Dict, zero_based: bool = True, anatomy: str = None, opt: Dict = None
) -> Optional[np.ndarray]:
    """Get triangle faces (default: 0-based indexing)."""
    container = _get_mesh_container(jgii, anatomy)
    if not container:
        return None
    try:
        faces = _decode_jdata(container["MeshTri3"]["Data"], opt, root=jgii)
        if isinstance(faces, dict):
            return faces
        return faces - 1 if zero_based else faces
    except KeyError:
        return None


def get_property(
    jgii: Dict, name: str, anatomy: str = None, opt: Dict = None
) -> Optional[np.ndarray]:
    """Get a named property from JGIFTI structure."""
    container = _get_mesh_container(jgii, anatomy)
    if not container:
        return None
    try:
        props = container["MeshVertex3"]["Properties"]
        if name in props:
            return _decode_jdata(props[name], opt, root=jgii)
        name_lower = name.lower()
        for k, v in props.items():
            if k.lower() == name_lower:
                return _decode_jdata(v, opt, root=jgii)
        return None
    except KeyError:
        return None


def get_properties(jgii: Dict, anatomy: str = None, opt: Dict = None) -> Optional[Dict]:
    """Get all properties dict (decoded)."""
    container = _get_mesh_container(jgii, anatomy)
    if not container:
        return None
    try:
        return _decode_properties(
            container["MeshVertex3"]["Properties"], opt, root=jgii
        )
    except KeyError:
        return None


def get_labels(jgii: Dict) -> Optional[Dict]:
    """Get label table from JGIFTI structure."""
    return jgii.get("GIFTIHeader", {}).get("LabelTable")


def get_metadata(
    jgii: Dict, level: str = "file", anatomy: str = None
) -> Optional[Dict]:
    """Get metadata ('file' or 'node' level)."""
    if level == "file":
        return jgii.get("GIFTIHeader", {}).get("MetaData")
    elif level == "node":
        container = _get_mesh_container(jgii, anatomy)
        if container:
            try:
                return container["MeshVertex3"]["_DataInfo_"]["MetaData"]
            except KeyError:
                pass
    return None


def get_coord_system(jgii: Dict, anatomy: str = None) -> Optional[Dict]:
    """Get coordinate system transform."""
    container = _get_mesh_container(jgii, anatomy)
    if container:
        try:
            return container["MeshVertex3"]["_DataInfo_"]["CoordSystem"]
        except KeyError:
            pass
    return jgii.get("GIFTIHeader", {}).get("CoordSystem")


def gii2jgii(filename: str, **kwargs) -> Dict:
    """Convert GIFTI (.gii) file to JGIFTI structure."""
    if filename.endswith(".gz"):
        import gzip

        with gzip.open(filename, "rb") as f:
            root = ET.fromstring(f.read())
    else:
        root = ET.parse(filename).getroot()

    header = {"Version": root.get("Version", "1.0")}

    file_meta = _parse_metadata(root.find("MetaData"))
    if file_meta:
        header["MetaData"] = file_meta

    labels = _parse_label_table(root.find("LabelTable"))
    if labels:
        header["LabelTable"] = labels

    gifti_data = {}
    node_data, node_info, properties = None, None, {}

    for da in root.findall("DataArray"):
        attribs = da.attrib
        intent = attribs.get("Intent", "NIFTI_INTENT_NONE")

        data_info = {}
        da_meta = _parse_metadata(da.find("MetaData"))
        if da_meta:
            data_info["MetaData"] = da_meta

        coord_systems = [
            cs
            for cs in (
                _parse_coord_system(c)
                for c in da.findall("CoordinateSystemTransformMatrix")
            )
            if cs
        ]
        if coord_systems:
            data_info["CoordSystem"] = (
                coord_systems[0] if len(coord_systems) == 1 else coord_systems
            )

        data = _decode_gifti_data(da.find("Data"), attribs)

        if intent == "NIFTI_INTENT_POINTSET":
            node_data, node_info = data, data_info
        elif intent == "NIFTI_INTENT_TRIANGLE":
            tri_data = {"Data": data + 1 if isinstance(data, np.ndarray) else data}
            if data_info:
                tri_data["_DataInfo_"] = data_info
            gifti_data["MeshTri3"] = tri_data
        else:
            prop_name = da_meta.get("Name") or _INTENT_PROPERTY.get(intent, "Data")
            properties[prop_name] = data

    if node_data is not None or properties:
        mesh_node = {}
        if node_info:
            mesh_node["_DataInfo_"] = node_info
        if node_data is not None:
            mesh_node["Data"] = node_data
        if properties:
            mesh_node["Properties"] = properties
        gifti_data["MeshVertex3"] = mesh_node

    return {"GIFTIHeader": header, "GIFTIData": gifti_data}


def jgii2gii(jgii: Dict, filename: Optional[str] = None, **kwargs) -> bytes:
    """Convert JGIFTI structure to GIFTI XML format."""
    encoding = kwargs.get("encoding", "GZipBase64Binary")
    header = jgii.get("GIFTIHeader", {})
    gifti_data = jgii.get("GIFTIData", {})

    if "MeshVertex3" not in gifti_data and "MeshTri3" not in gifti_data:
        for key, val in gifti_data.items():
            if isinstance(val, dict) and ("MeshVertex3" in val or "MeshTri3" in val):
                gifti_data = val
                break

    num_arrays = 0
    if "MeshVertex3" in gifti_data:
        mv = gifti_data["MeshVertex3"]
        if "Data" in mv and mv["Data"] is not None:
            num_arrays += 1
        num_arrays += len(mv.get("Properties", {}))
    if "MeshTri3" in gifti_data:
        num_arrays += 1

    root = ET.Element(
        "GIFTI",
        Version=header.get("Version", "1.0"),
        NumberOfDataArrays=str(num_arrays),
    )

    if header.get("MetaData"):
        meta_elem = ET.SubElement(root, "MetaData")
        for name, value in header["MetaData"].items():
            md = ET.SubElement(meta_elem, "MD")
            ET.SubElement(md, "Name").text = name
            ET.SubElement(md, "Value").text = str(value)

    if header.get("LabelTable"):
        label_elem = ET.SubElement(root, "LabelTable")
        for key, ldata in header["LabelTable"].items():
            rgba = ldata.get("RGBA", [0.667, 0.667, 0.667, 1.0])
            label = ET.SubElement(
                label_elem,
                "Label",
                Key=str(key),
                Red=f"{rgba[0]:.3f}",
                Green=f"{rgba[1]:.3f}",
                Blue=f"{rgba[2]:.3f}",
                Alpha=f"{rgba[3]:.3f}",
            )
            label.text = ldata.get("Label", "???")

    def add_data_array(data, intent: str, data_info: Dict = None):
        """Helper to add a DataArray element."""
        data = _decode_jdata(data, root=jgii)
        if isinstance(data, dict):
            return

        data = np.asarray(data)
        dtype_name = _DATATYPE_MAP_REV.get(str(data.dtype), "NIFTI_TYPE_FLOAT32")

        attribs = {
            "Intent": intent,
            "DataType": dtype_name,
            "ArrayIndexingOrder": "RowMajorOrder",
            "Dimensionality": str(data.ndim),
            "Encoding": encoding,
            "Endian": "LittleEndian",
        }
        for i, d in enumerate(data.shape):
            attribs[f"Dim{i}"] = str(d)

        da = ET.SubElement(root, "DataArray", **attribs)

        if data_info and data_info.get("MetaData"):
            meta_elem = ET.SubElement(da, "MetaData")
            for name, value in data_info["MetaData"].items():
                md = ET.SubElement(meta_elem, "MD")
                ET.SubElement(md, "Name").text = name
                ET.SubElement(md, "Value").text = str(value)

        if data_info and data_info.get("CoordSystem"):
            cs_list = data_info["CoordSystem"]
            if not isinstance(cs_list, list):
                cs_list = [cs_list]
            for cs in cs_list:
                cs_elem = ET.SubElement(da, "CoordinateSystemTransformMatrix")
                ds = cs.get("DataSpace", "talairach")
                ET.SubElement(cs_elem, "DataSpace").text = _XFORM_MAP_REV.get(
                    ds, f"NIFTI_XFORM_{ds.upper()}"
                )
                ts = cs.get("TransformedSpace", "talairach")
                ET.SubElement(cs_elem, "TransformedSpace").text = _XFORM_MAP_REV.get(
                    ts, f"NIFTI_XFORM_{ts.upper()}"
                )
                mat = cs.get("MatrixData", _IDENTITY_MATRIX)
                ET.SubElement(cs_elem, "MatrixData").text = " ".join(
                    str(v) for row in mat for v in row
                )

        data_elem = ET.SubElement(da, "Data")
        flat = data.flatten(order="C")
        if encoding == "ASCII":
            data_elem.text = " ".join(map(str, flat))
        elif encoding == "Base64Binary":
            data_elem.text = base64.b64encode(flat.tobytes()).decode("ascii")
        else:
            data_elem.text = base64.b64encode(zlib.compress(flat.tobytes())).decode(
                "ascii"
            )

    if "MeshVertex3" in gifti_data:
        mv = gifti_data["MeshVertex3"]
        if mv.get("Data") is not None:
            add_data_array(mv["Data"], "NIFTI_INTENT_POINTSET", mv.get("_DataInfo_"))

        for prop_name, prop_data in mv.get("Properties", {}).items():
            prop_lower = prop_name.lower()
            intent = _PROPERTY_INTENT.get(prop_lower, "NIFTI_INTENT_NONE")

            decoded = _decode_jdata(prop_data, root=jgii)
            if isinstance(decoded, dict):
                continue
            decoded = np.asarray(decoded)

            if prop_lower == "color":
                intent = (
                    "NIFTI_INTENT_RGBA_VECTOR"
                    if decoded.shape[-1] == 4
                    else "NIFTI_INTENT_RGB_VECTOR"
                )
            if prop_lower == "label":
                decoded = decoded.astype(np.int32)
            add_data_array(decoded, intent, {"MetaData": {"Name": prop_name}})

    if "MeshTri3" in gifti_data:
        mt = gifti_data["MeshTri3"]
        data = _decode_jdata(mt["Data"], root=jgii)
        if not isinstance(data, dict):
            data = np.asarray(data, dtype=np.int32) - 1
            add_data_array(data, "NIFTI_INTENT_TRIANGLE", mt.get("_DataInfo_"))

    xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="utf-8"
    )

    if filename:
        if filename.endswith(".gz"):
            import gzip

            with gzip.open(filename, "wb") as f:
                f.write(xml_bytes)
        else:
            with open(filename, "wb") as f:
                f.write(xml_bytes)

    return xml_bytes


def loadgifti(filename: str, **kwargs) -> Dict:
    """Load a GIFTI file and return as JGIFTI structure."""
    return gii2jgii(filename, **kwargs)


def loadjgifti(filename: str, opt: Dict = None, **kwargs) -> Dict:
    """Load a JGIFTI (.jgii/.bgii) or GIFTI (.gii) file."""
    import jdata as jd

    opt = opt or {}
    opt.update(kwargs)

    fl = filename.lower()
    if fl.endswith(".gii") or fl.endswith(".gii.gz"):
        return gii2jgii(filename, **kwargs)
    elif fl.endswith(".jgii") or fl.endswith(".bgii"):
        return jd.load(filename, opt, **kwargs)
    raise ValueError("File must be .gii, .jgii, or .bgii")


def savegifti(jgii: Dict, filename: str, **kwargs):
    """Save JGIFTI structure to GIFTI XML file."""
    jgii2gii(jgii, filename, **kwargs)


def savejgifti(jgii: Dict, filename: str, **kwargs):
    """Save JGIFTI structure to file (.jgii, .bgii, or .gii)."""
    import jdata as jd

    fl = filename.lower()
    if fl.endswith(".jgii") or fl.endswith(".bgii"):
        jd.save(jgii, filename, **kwargs)
    elif fl.endswith(".gii") or fl.endswith(".gii.gz"):
        jgii2gii(jgii, filename, **kwargs)
    else:
        raise ValueError("File suffix must be .jgii, .bgii, or .gii")


def jgifticreate(
    node: np.ndarray = None, face: np.ndarray = None, properties: Dict = None, **kwargs
) -> Dict:
    """Create a JGIFTI structure with optional mesh data."""
    jgii = {
        "_DataInfo_": {
            "JGIFTIVersion": "1.0",
            "Comment": "Created by JGIFTI Python module",
            "AnnotationFormat": "https://github.com/NeuroJSON/jgifti",
        },
        "GIFTIHeader": {"Version": "1.0", "MetaData": {}},
    }

    if node is None and face is None:
        return jgii

    gifti_data = {}

    if node is not None:
        gifti_data["MeshVertex3"] = {
            "_DataInfo_": {
                "MetaData": {},
                "CoordSystem": {
                    "DataSpace": "talairach",
                    "TransformedSpace": "talairach",
                    "MatrixData": _IDENTITY_MATRIX,
                },
            },
            "Data": np.asarray(node, dtype=np.float32),
            "Properties": dict(properties) if properties else {},
        }

    if face is not None:
        gifti_data["MeshTri3"] = {
            "_DataInfo_": {"MetaData": {"TopologicalType": "Closed"}},
            "Data": np.asarray(face, dtype=np.int32) + 1,
        }

    jgii["GIFTIData"] = gifti_data
    return jgii


# ============================================================================
# JGifti Class
# ============================================================================


class JGifti:
    """
    A class to manage JGIFTI surface mesh data.

    Provides methods to load/save GIFTI and JGIFTI files, and access
    mesh geometry, properties, labels, coordinate systems, and metadata.

    Supports both single-surface and multi-surface files with anatomy names.

    Examples
    --------
    >>> # Create from file
    >>> gii = JGifti("brain.gii")
    >>> nodes = gii.node()
    >>> faces = gii.face()
    >>> thickness = gii["Thickness"]

    >>> # Create from data
    >>> gii = JGifti(node=verts, face=tris)
    >>> gii["Curvature"] = curv_data
    >>> gii.save("output.jgii")

    >>> # Multi-surface file
    >>> gii = JGifti("multi_surface.jgii")
    >>> print(gii.surfaces)
    >>> pial_nodes = gii.node("P001_L_pial")

    >>> # Create with anatomy name
    >>> gii = JGifti(node=verts, face=tris, anatomy="P001_L_pial")
    """

    def __init__(
        self,
        filename: str = None,
        node: np.ndarray = None,
        face: np.ndarray = None,
        properties: Dict = None,
        anatomy: str = None,
        **kwargs,
    ):
        """
        Initialize JGifti object.

        Parameters
        ----------
        filename : str, optional
            Path to GIFTI (.gii) or JGIFTI (.jgii, .bgii) file to load
        node : np.ndarray, optional
            Nx3 array of vertex/node coordinates (used if filename not provided)
        face : np.ndarray, optional
            Mx3 array of triangle indices, 0-based (used if filename not provided)
        properties : dict, optional
            Dictionary of node properties
        anatomy : str, optional
            Surface identifier for multi-surface structure (e.g., 'P001_L_pial')
        **kwargs
            Additional options passed to loader
        """
        self._data = None
        self._opt = kwargs
        self._default_anatomy = anatomy

        if filename:
            self.load(filename, **kwargs)
        elif node is not None or face is not None:
            self._data = self._create_structure(node, face, properties, anatomy)
        else:
            self._data = jgifticreate()

    def _create_structure(
        self,
        node: np.ndarray = None,
        face: np.ndarray = None,
        properties: Dict = None,
        anatomy: str = None,
    ) -> Dict:
        """Create JGIFTI structure, optionally nested under anatomy name."""
        base = jgifticreate(node, face, properties)

        if anatomy and "GIFTIData" in base:
            mesh_data = base["GIFTIData"]
            base["GIFTIData"] = {anatomy: mesh_data}

        return base

    @property
    def data(self) -> Dict:
        """Return the underlying JGIFTI dictionary structure."""
        return self._data

    @data.setter
    def data(self, value: Dict):
        """Set the underlying JGIFTI dictionary structure."""
        self._data = value

    # ========================================================================
    # File I/O
    # ========================================================================

    def load(self, filename: str, **kwargs) -> "JGifti":
        """
        Load data from a GIFTI or JGIFTI file.

        Parameters
        ----------
        filename : str
            Path to .gii, .gii.gz, .jgii, or .bgii file
        **kwargs
            Additional options passed to loader

        Returns
        -------
        self : JGifti
            Returns self for method chaining
        """
        self._data = loadjgifti(filename, **kwargs)
        return self

    def save(self, filename: str, **kwargs) -> "JGifti":
        """
        Save data to a GIFTI or JGIFTI file.

        Parameters
        ----------
        filename : str
            Output path (.gii, .gii.gz, .jgii, or .bgii)
        **kwargs
            encoding : str
                For .gii files: 'ASCII', 'Base64Binary', or 'GZipBase64Binary'
            compression : str
                For .jgii/.bgii files: 'zlib', 'lzma', etc.

        Returns
        -------
        self : JGifti
            Returns self for method chaining
        """
        savejgifti(self._data, filename, **kwargs)
        return self

    # ========================================================================
    # Surface identification (for multi-surface files)
    # ========================================================================

    @property
    def surfaces(self) -> list:
        """
        Get list of surface names (anatomy identifiers).

        Returns empty list for single-surface (direct) structure.
        """
        return get_surfaces(self._data)

    @property
    def is_multi_surface(self) -> bool:
        """Check if this is a multi-surface file."""
        return len(self.surfaces) > 0

    @property
    def anatomy(self) -> Optional[str]:
        """Get/set default anatomy name for property access."""
        return self._default_anatomy

    @anatomy.setter
    def anatomy(self, value: str):
        """Set default anatomy name."""
        self._default_anatomy = value

    def _resolve_anatomy(self, anatomy: str = None) -> Optional[str]:
        """Resolve anatomy, using default if not specified."""
        return anatomy if anatomy is not None else self._default_anatomy

    # ========================================================================
    # Geometry access
    # ========================================================================

    def node(self, anatomy: str = None) -> Optional[np.ndarray]:
        """
        Get node/vertex coordinates (Nx3).

        Parameters
        ----------
        anatomy : str, optional
            Surface identifier. Uses default if not specified.
        """
        anat = self._resolve_anatomy(anatomy)
        return get_node(self._data, anat, opt=self._opt)

    def face(
        self, anatomy: str = None, zero_based: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get triangle faces (Mx3).

        Parameters
        ----------
        anatomy : str, optional
            Surface identifier
        zero_based : bool
            If True (default), return 0-based indices
        """
        anat = self._resolve_anatomy(anatomy)
        return get_face(self._data, zero_based, anat, opt=self._opt)

    def nnode(self, anatomy: str = None) -> int:
        """Get number of nodes/vertices."""
        n = self.node(anatomy)
        return n.shape[0] if n is not None else 0

    def nface(self, anatomy: str = None) -> int:
        """Get number of faces."""
        f = self.face(anatomy)
        return f.shape[0] if f is not None else 0

    # ========================================================================
    # Property access (nodal values)
    # ========================================================================

    def __getitem__(self, name: str) -> Optional[np.ndarray]:
        """
        Get a node property by name (from default surface).

        Parameters
        ----------
        name : str
            Property name (e.g., 'Thickness', 'Curvature', 'Label')

        Returns
        -------
        np.ndarray or None
            Property values, or None if not found
        """
        return self.get_property(name)

    def __setitem__(self, name: str, value):
        """
        Set a node property (on default surface).

        Parameters
        ----------
        name : str
            Property name
        value : array-like
            Property values (length must match number of nodes)
        """
        self.set_property(name, value)

    def __contains__(self, name: str) -> bool:
        """Check if a property exists."""
        return self.get_property(name) is not None

    def get_property(self, name: str, anatomy: str = None) -> Optional[np.ndarray]:
        """
        Get a node property.

        Parameters
        ----------
        name : str
            Property name
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        anat = self._resolve_anatomy(anatomy)
        return get_property(self._data, name, anat, opt=self._opt)

    def set_property(self, name: str, value, anatomy: str = None):
        """
        Set a node property.

        Parameters
        ----------
        name : str
            Property name
        value : array-like
            Property values
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        anat = self._resolve_anatomy(anatomy)
        container = _get_mesh_container(self._data, anat)

        if container is None:
            raise ValueError("No mesh data exists. Add nodes first.")

        if "MeshVertex3" not in container:
            container["MeshVertex3"] = {"Properties": {}}
        if "Properties" not in container["MeshVertex3"]:
            container["MeshVertex3"]["Properties"] = {}

        container["MeshVertex3"]["Properties"][name] = np.asarray(value)

    def properties(self, anatomy: str = None) -> Optional[Dict]:
        """Get all properties as a dictionary (decoded)."""
        anat = self._resolve_anatomy(anatomy)
        return get_properties(self._data, anat, opt=self._opt)

    def property_names(self, anatomy: str = None) -> list:
        """Get list of available property names."""
        anat = self._resolve_anatomy(anatomy)
        container = _get_mesh_container(self._data, anat)
        if container:
            try:
                return list(container["MeshVertex3"]["Properties"].keys())
            except KeyError:
                pass
        return []

    # ========================================================================
    # Labels
    # ========================================================================

    @property
    def labels(self) -> Optional[Dict]:
        """
        Get label table.

        Returns
        -------
        dict or None
            Dictionary mapping label keys to {Label, RGBA} dicts
        """
        return get_labels(self._data)

    @labels.setter
    def labels(self, value: Dict):
        """Set label table."""
        if "GIFTIHeader" not in self._data:
            self._data["GIFTIHeader"] = {}
        self._data["GIFTIHeader"]["LabelTable"] = value

    def get_label_name(self, key: Union[int, str]) -> Optional[str]:
        """Get label name for a given key."""
        labels = self.labels
        if labels:
            key_str = str(key)
            if key_str in labels:
                return labels[key_str].get("Label")
        return None

    def get_label_color(self, key: Union[int, str]) -> Optional[list]:
        """Get RGBA color for a label key."""
        labels = self.labels
        if labels:
            key_str = str(key)
            if key_str in labels:
                return labels[key_str].get("RGBA")
        return None

    def add_label(self, key: Union[int, str], name: str, rgba: list = None):
        """
        Add or update a label in the label table.

        Parameters
        ----------
        key : int or str
            Label key/index
        name : str
            Label name
        rgba : list, optional
            [R, G, B, A] color values (0.0-1.0), default gray
        """
        if self.labels is None:
            self.labels = {}
        if rgba is None:
            rgba = [0.667, 0.667, 0.667, 1.0]
        self._data["GIFTIHeader"]["LabelTable"][str(key)] = {
            "Label": name,
            "RGBA": rgba,
        }

    # ========================================================================
    # Coordinate system
    # ========================================================================

    def coord_system(self, anatomy: str = None) -> Optional[Dict]:
        """Get coordinate system transform."""
        anat = self._resolve_anatomy(anatomy)
        return get_coord_system(self._data, anat)

    def data_space(self, anatomy: str = None) -> Optional[str]:
        """Get data space name."""
        cs = self.coord_system(anatomy)
        return cs.get("DataSpace") if cs else None

    def transformed_space(self, anatomy: str = None) -> Optional[str]:
        """Get transformed space name."""
        cs = self.coord_system(anatomy)
        return cs.get("TransformedSpace") if cs else None

    def transform_matrix(self, anatomy: str = None) -> Optional[np.ndarray]:
        """Get 4x4 transformation matrix as numpy array."""
        cs = self.coord_system(anatomy)
        if cs and "MatrixData" in cs:
            return np.array(cs["MatrixData"])
        return None

    def set_coord_system(
        self,
        data_space: str = "talairach",
        transformed_space: str = "talairach",
        matrix: np.ndarray = None,
        anatomy: str = None,
    ):
        """
        Set coordinate system transform.

        Parameters
        ----------
        data_space : str
            Source space name
        transformed_space : str
            Target space name
        matrix : np.ndarray, optional
            4x4 transformation matrix (default: identity)
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        if matrix is None:
            matrix = _IDENTITY_MATRIX
        else:
            matrix = np.asarray(matrix).tolist()

        cs = {
            "DataSpace": data_space,
            "TransformedSpace": transformed_space,
            "MatrixData": matrix,
        }

        anat = self._resolve_anatomy(anatomy)
        container = _get_mesh_container(self._data, anat)
        if container and "MeshVertex3" in container:
            if "_DataInfo_" not in container["MeshVertex3"]:
                container["MeshVertex3"]["_DataInfo_"] = {}
            container["MeshVertex3"]["_DataInfo_"]["CoordSystem"] = cs

    # ========================================================================
    # Metadata
    # ========================================================================

    @property
    def metadata(self) -> Optional[Dict]:
        """Get file-level metadata."""
        return get_metadata(self._data, "file")

    @metadata.setter
    def metadata(self, value: Dict):
        """Set file-level metadata."""
        if "GIFTIHeader" not in self._data:
            self._data["GIFTIHeader"] = {}
        self._data["GIFTIHeader"]["MetaData"] = value

    def get_metadata(self, level: str = "file", anatomy: str = None) -> Optional[Dict]:
        """
        Get metadata at specified level.

        Parameters
        ----------
        level : str
            'file' for header metadata, 'node' for MeshVertex3 metadata
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        anat = self._resolve_anatomy(anatomy) if level == "node" else None
        return get_metadata(self._data, level, anat)

    def set_metadata(
        self, key: str, value: str, level: str = "file", anatomy: str = None
    ):
        """
        Set a metadata value.

        Parameters
        ----------
        key : str
            Metadata key name
        value : str
            Metadata value
        level : str
            'file' or 'node'
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        if level == "file":
            if "GIFTIHeader" not in self._data:
                self._data["GIFTIHeader"] = {}
            if "MetaData" not in self._data["GIFTIHeader"]:
                self._data["GIFTIHeader"]["MetaData"] = {}
            self._data["GIFTIHeader"]["MetaData"][key] = value
        elif level == "node":
            anat = self._resolve_anatomy(anatomy)
            container = _get_mesh_container(self._data, anat)
            if container and "MeshVertex3" in container:
                mv = container["MeshVertex3"]
                if "_DataInfo_" not in mv:
                    mv["_DataInfo_"] = {}
                if "MetaData" not in mv["_DataInfo_"]:
                    mv["_DataInfo_"]["MetaData"] = {}
                mv["_DataInfo_"]["MetaData"][key] = value

    @property
    def version(self) -> str:
        """Get GIFTI format version."""
        return self._data.get("GIFTIHeader", {}).get("Version", "1.0")

    # ========================================================================
    # Mesh manipulation
    # ========================================================================

    def set_node(self, node: np.ndarray, anatomy: str = None):
        """
        Set node/vertex coordinates.

        Parameters
        ----------
        node : np.ndarray
            Nx3 array of node coordinates
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        node = np.asarray(node, dtype=np.float32)
        anat = self._resolve_anatomy(anatomy)

        if "GIFTIData" not in self._data:
            self._data["GIFTIData"] = {}

        gifti_data = self._data["GIFTIData"]

        if anat:
            if anat not in gifti_data:
                gifti_data[anat] = {}
            container = gifti_data[anat]
        else:
            if "MeshVertex3" in gifti_data or "MeshTri3" in gifti_data:
                container = gifti_data
            elif self.is_multi_surface and self.surfaces:
                container = gifti_data[self.surfaces[0]]
            else:
                container = gifti_data

        if "MeshVertex3" not in container:
            container["MeshVertex3"] = {
                "_DataInfo_": {
                    "MetaData": {},
                    "CoordSystem": {
                        "DataSpace": "talairach",
                        "TransformedSpace": "talairach",
                        "MatrixData": _IDENTITY_MATRIX,
                    },
                },
                "Properties": {},
            }
        container["MeshVertex3"]["Data"] = node

    def set_face(self, face: np.ndarray, zero_based: bool = True, anatomy: str = None):
        """
        Set triangle faces.

        Parameters
        ----------
        face : np.ndarray
            Mx3 array of triangle indices
        zero_based : bool
            If True (default), input is 0-based and will be converted to 1-based
        anatomy : str, optional
            Surface identifier for multi-surface files
        """
        face = np.asarray(face, dtype=np.int32)
        if zero_based:
            face = face + 1

        anat = self._resolve_anatomy(anatomy)

        if "GIFTIData" not in self._data:
            self._data["GIFTIData"] = {}

        gifti_data = self._data["GIFTIData"]

        if anat:
            if anat not in gifti_data:
                gifti_data[anat] = {}
            container = gifti_data[anat]
        else:
            if "MeshVertex3" in gifti_data or "MeshTri3" in gifti_data:
                container = gifti_data
            elif self.is_multi_surface and self.surfaces:
                container = gifti_data[self.surfaces[0]]
            else:
                container = gifti_data

        if "MeshTri3" not in container:
            container["MeshTri3"] = {
                "_DataInfo_": {"MetaData": {"TopologicalType": "Closed"}}
            }
        container["MeshTri3"]["Data"] = face

    def add_surface(
        self,
        anatomy: str,
        node: np.ndarray = None,
        face: np.ndarray = None,
        properties: Dict = None,
        share_topology_with: str = None,
    ):
        """
        Add a new surface to a multi-surface structure.

        Parameters
        ----------
        anatomy : str
            Surface identifier (e.g., 'P001_L_white')
        node : np.ndarray, optional
            Nx3 node coordinates
        face : np.ndarray, optional
            Mx3 triangle indices (0-based)
        properties : dict, optional
            Node properties
        share_topology_with : str, optional
            If specified, use _DataLink_ to reference another surface's topology
        """
        if "GIFTIData" not in self._data:
            self._data["GIFTIData"] = {}

        gifti_data = self._data["GIFTIData"]

        if "MeshVertex3" in gifti_data or "MeshTri3" in gifti_data:
            existing = {
                k: v for k, v in gifti_data.items() if k in ("MeshVertex3", "MeshTri3")
            }
            for k in existing:
                del gifti_data[k]
            gifti_data["default"] = existing

        surface = {}

        if node is not None:
            node = np.asarray(node, dtype=np.float32)
            surface["MeshVertex3"] = {
                "_DataInfo_": {
                    "MetaData": {},
                    "CoordSystem": {
                        "DataSpace": "talairach",
                        "TransformedSpace": "talairach",
                        "MatrixData": _IDENTITY_MATRIX,
                    },
                },
                "Data": node,
                "Properties": dict(properties) if properties else {},
            }

        if face is not None:
            face = np.asarray(face, dtype=np.int32) + 1
            surface["MeshTri3"] = {
                "_DataInfo_": {"MetaData": {"TopologicalType": "Closed"}},
                "Data": face,
            }
        elif share_topology_with:
            surface["MeshTri3"] = {
                "_DataInfo_": {},
                "Data": {
                    "_DataLink_": f"$.GIFTIData.{share_topology_with}.MeshTri3.Data"
                },
            }

        gifti_data[anatomy] = surface

    # ========================================================================
    # Utility methods
    # ========================================================================

    def copy(self) -> "JGifti":
        """Create a deep copy of this JGifti object."""
        import copy

        new_obj = JGifti()
        new_obj._data = copy.deepcopy(self._data)
        new_obj._opt = self._opt.copy()
        new_obj._default_anatomy = self._default_anatomy
        return new_obj

    def __repr__(self) -> str:
        """String representation."""
        surfaces = self.surfaces
        if surfaces:
            info = f"surfaces={surfaces}"
        else:
            nn, nf = self.nnode(), self.nface()
            info = f"node={nn}, face={nf}"

        props = self.property_names()
        if props:
            info += f", properties={props}"

        return f"JGifti({info})"

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = ["JGifti Surface Mesh"]
        lines.append(f"  Version: {self.version}")

        surfaces = self.surfaces
        if surfaces:
            lines.append(f"  Surfaces: {surfaces}")
            for surf in surfaces:
                nn = self.nnode(surf)
                nf = self.nface(surf)
                lines.append(f"    {surf}: {nn} nodes, {nf} faces")
        else:
            lines.append(f"  Nodes: {self.nnode()}")
            lines.append(f"  Faces: {self.nface()}")

        props = self.property_names()
        if props:
            lines.append(f"  Properties: {props}")

        labels = self.labels
        if labels:
            lines.append(f"  Labels: {len(labels)} entries")

        cs = self.coord_system()
        if cs:
            lines.append(
                f"  Coord System: {cs.get('DataSpace')} -> {cs.get('TransformedSpace')}"
            )

        meta = self.metadata
        if meta:
            lines.append(f"  Metadata: {list(meta.keys())}")

        return "\n".join(lines)
