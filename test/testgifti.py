"""
Unit tests for JGIFTI module

Run with: python -m unittest test_jgifti -v
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import the module (adjust path as needed)
from jdata.jgifti import (
    JGifti,
    gii2jgii,
    jgii2gii,
    loadgifti,
    loadjgifti,
    savegifti,
    savejgifti,
    jgifticreate,
    giicodemap,
    get_node,
    get_face,
    get_property,
    get_properties,
    get_labels,
    get_metadata,
    get_coord_system,
    get_surfaces,
)


class TestJGIFTICreate(unittest.TestCase):
    """Tests for JGIFTI structure creation."""

    def test_jgifticreate_empty(self):
        """Test creating empty JGIFTI structure."""
        jgii = jgifticreate()
        self.assertIn("GIFTIHeader", jgii)
        self.assertEqual(jgii["GIFTIHeader"]["Version"], "1.0")

    def test_jgifticreate_with_node(self):
        """Test creating JGIFTI with node data."""
        node = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        jgii = jgifticreate(node=node)

        self.assertIn("GIFTIData", jgii)
        self.assertIn("MeshVertex3", jgii["GIFTIData"])
        np.testing.assert_array_equal(jgii["GIFTIData"]["MeshVertex3"]["Data"], node)

    def test_jgifticreate_with_node_and_face(self):
        """Test creating JGIFTI with nodes and faces."""
        node = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        face = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3],
                [0, 2, 3],
            ],
            dtype=np.int32,
        )

        jgii = jgifticreate(node=node, face=face)

        self.assertIn("MeshVertex3", jgii["GIFTIData"])
        self.assertIn("MeshTri3", jgii["GIFTIData"])

        # Faces should be stored as 1-based
        stored_face = jgii["GIFTIData"]["MeshTri3"]["Data"]
        np.testing.assert_array_equal(stored_face, face + 1)

    def test_jgifticreate_with_properties(self):
        """Test creating JGIFTI with node properties."""
        node = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        properties = {
            "Thickness": np.array([2.5, 2.3, 2.8]),
            "Curvature": np.array([0.1, -0.2, 0.05]),
        }

        jgii = jgifticreate(node=node, properties=properties)

        props = jgii["GIFTIData"]["MeshVertex3"]["Properties"]
        self.assertIn("Thickness", props)
        self.assertIn("Curvature", props)
        np.testing.assert_array_almost_equal(props["Thickness"], [2.5, 2.3, 2.8])


class TestAccessorFunctions(unittest.TestCase):
    """Tests for JGIFTI data accessor functions."""

    def setUp(self):
        """Create a sample JGIFTI structure for testing."""
        node = np.array(
            [
                [-16.07, -66.19, 21.27],
                [-16.71, -66.05, 21.23],
                [-17.61, -65.40, 21.07],
            ],
            dtype=np.float32,
        )

        face = np.array([[0, 1, 2]], dtype=np.int32)

        properties = {
            "Thickness": np.array([2.5, 2.3, 2.8]),
            "Label": np.array([0, 1, 1], dtype=np.int32),
        }

        self.jgii = jgifticreate(node=node, face=face, properties=properties)

        # Add label table
        self.jgii["GIFTIHeader"]["LabelTable"] = {
            "0": {"Label": "???", "RGBA": [0.667, 0.667, 0.667, 1.0]},
            "1": {"Label": "V1", "RGBA": [1.0, 0.0, 0.0, 1.0]},
        }

    def test_get_node(self):
        """Test getting nodes."""
        nodes = get_node(self.jgii)
        self.assertIsNotNone(nodes)
        self.assertEqual(nodes.shape, (3, 3))

    def test_get_face_zero_based(self):
        """Test getting faces with 0-based indexing."""
        faces = get_face(self.jgii, zero_based=True)
        self.assertIsNotNone(faces)
        self.assertEqual(faces.shape, (1, 3))
        self.assertEqual(faces.min(), 0)

    def test_get_face_one_based(self):
        """Test getting faces with 1-based indexing."""
        faces = get_face(self.jgii, zero_based=False)
        self.assertIsNotNone(faces)
        self.assertEqual(faces.min(), 1)

    def test_get_property(self):
        """Test getting a named property."""
        thickness = get_property(self.jgii, "Thickness")
        self.assertIsNotNone(thickness)
        np.testing.assert_array_almost_equal(thickness, [2.5, 2.3, 2.8])

    def test_get_property_case_insensitive(self):
        """Test case-insensitive property lookup."""
        thickness = get_property(self.jgii, "thickness")
        self.assertIsNotNone(thickness)

    def test_get_labels(self):
        """Test getting label table."""
        labels = get_labels(self.jgii)
        self.assertIsNotNone(labels)
        self.assertIn("0", labels)
        self.assertEqual(labels["1"]["Label"], "V1")

    def test_get_surfaces_single(self):
        """Test get_surfaces returns empty list for single surface."""
        surfaces = get_surfaces(self.jgii)
        self.assertEqual(surfaces, [])


class TestMultiSurface(unittest.TestCase):
    """Tests for multi-surface JGIFTI structures."""

    def setUp(self):
        """Create a multi-surface JGIFTI structure."""
        self.jgii = {
            "GIFTIHeader": {"Version": "1.0", "MetaData": {}},
            "GIFTIData": {
                "P001_L_pial": {
                    "MeshVertex3": {
                        "_DataInfo_": {"MetaData": {}},
                        "Data": np.array(
                            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
                        ),
                        "Properties": {"Thickness": np.array([2.5, 2.3, 2.8])},
                    },
                    "MeshTri3": {
                        "_DataInfo_": {},
                        "Data": np.array([[1, 2, 3]], dtype=np.int32),
                    },
                },
                "P001_L_white": {
                    "MeshVertex3": {
                        "_DataInfo_": {"MetaData": {}},
                        "Data": np.array(
                            [[0, 0, -1], [1, 0, -1], [0, 1, -1]], dtype=np.float32
                        ),
                        "Properties": {},
                    },
                    "MeshTri3": {
                        "_DataInfo_": {},
                        "Data": {"_DataLink_": "$.GIFTIData.P001_L_pial.MeshTri3.Data"},
                    },
                },
            },
        }

    def test_get_surfaces(self):
        """Test getting list of surfaces."""
        surfaces = get_surfaces(self.jgii)
        self.assertEqual(len(surfaces), 2)
        self.assertIn("P001_L_pial", surfaces)
        self.assertIn("P001_L_white", surfaces)

    def test_get_node_by_anatomy(self):
        """Test getting nodes for specific anatomy."""
        pial_nodes = get_node(self.jgii, anatomy="P001_L_pial")
        white_nodes = get_node(self.jgii, anatomy="P001_L_white")

        self.assertIsNotNone(pial_nodes)
        self.assertIsNotNone(white_nodes)
        self.assertEqual(pial_nodes[0, 2], 0)  # pial z=0
        self.assertEqual(white_nodes[0, 2], -1)  # white z=-1

    def test_get_property_by_anatomy(self):
        """Test getting property for specific anatomy."""
        thickness = get_property(self.jgii, "Thickness", anatomy="P001_L_pial")
        self.assertIsNotNone(thickness)

        # White surface has no thickness
        thickness_white = get_property(self.jgii, "Thickness", anatomy="P001_L_white")
        self.assertIsNone(thickness_white)

    def test_datalink_resolution(self):
        """Test that internal _DataLink_ references are resolved."""
        pial_faces = get_face(self.jgii, anatomy="P001_L_pial")
        white_faces = get_face(self.jgii, anatomy="P001_L_white")

        self.assertIsNotNone(pial_faces)
        self.assertIsNotNone(white_faces)
        np.testing.assert_array_equal(pial_faces, white_faces)


class TestCodeMapping(unittest.TestCase):
    """Tests for code mapping functions."""

    def test_xform_mapping(self):
        """Test XForm code mapping."""
        result = giicodemap("xform", "NIFTI_XFORM_TALAIRACH")
        self.assertEqual(result, "talairach")

    def test_datatype_mapping(self):
        """Test datatype mapping."""
        result = giicodemap("datatype", "NIFTI_TYPE_FLOAT32")
        self.assertEqual(result, "float32")


class TestGIFTIXMLParsing(unittest.TestCase):
    """Tests for GIFTI XML parsing."""

    def setUp(self):
        """Create a sample GIFTI XML file for testing."""
        self.tmp_dir = tempfile.mkdtemp()

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/1594/gifti.dtd">
<GIFTI Version="1.0" NumberOfDataArrays="2">
<MetaData>
<MD>
<Name>date</Name>
<Value>Thu Nov 15 09:05:22 2007</Value>
</MD>
</MetaData>
<LabelTable>
<Label Key="0" Red="0.667" Green="0.667" Blue="0.667" Alpha="1.0">???</Label>
<Label Key="1" Red="1.0" Green="0.0" Blue="0.0" Alpha="1.0">V1</Label>
</LabelTable>
<DataArray Intent="NIFTI_INTENT_POINTSET"
    DataType="NIFTI_TYPE_FLOAT32"
    ArrayIndexingOrder="RowMajorOrder"
    Dimensionality="2"
    Dim0="3"
    Dim1="3"
    Encoding="ASCII"
    Endian="LittleEndian">
<MetaData>
<MD>
<Name>AnatomicalStructurePrimary</Name>
<Value>CortexLeft</Value>
</MD>
</MetaData>
<CoordinateSystemTransformMatrix>
<DataSpace>NIFTI_XFORM_TALAIRACH</DataSpace>
<TransformedSpace>NIFTI_XFORM_TALAIRACH</TransformedSpace>
<MatrixData>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</MatrixData>
</CoordinateSystemTransformMatrix>
<Data>
-16.072010 -66.187515 21.266994
-16.705893 -66.054337 21.232786
-17.614349 -65.401642 21.071466
</Data>
</DataArray>
<DataArray Intent="NIFTI_INTENT_TRIANGLE"
    DataType="NIFTI_TYPE_INT32"
    ArrayIndexingOrder="RowMajorOrder"
    Dimensionality="2"
    Dim0="1"
    Dim1="3"
    Encoding="ASCII"
    Endian="LittleEndian">
<Data>0 1 2</Data>
</DataArray>
</GIFTI>"""

        self.gii_path = os.path.join(self.tmp_dir, "test.gii")
        with open(self.gii_path, "w") as f:
            f.write(xml_content)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir)

    def test_parse_gifti_basic(self):
        """Test basic GIFTI parsing."""
        jgii = gii2jgii(self.gii_path)

        self.assertIn("GIFTIHeader", jgii)
        self.assertIn("GIFTIData", jgii)
        self.assertEqual(jgii["GIFTIHeader"]["Version"], "1.0")

    def test_parse_gifti_metadata(self):
        """Test parsing file-level metadata."""
        jgii = gii2jgii(self.gii_path)

        meta = jgii["GIFTIHeader"]["MetaData"]
        self.assertIn("date", meta)

    def test_parse_gifti_labels(self):
        """Test parsing label table."""
        jgii = gii2jgii(self.gii_path)

        labels = jgii["GIFTIHeader"]["LabelTable"]
        self.assertIn("0", labels)
        self.assertEqual(labels["1"]["Label"], "V1")
        self.assertEqual(labels["1"]["RGBA"], [1.0, 0.0, 0.0, 1.0])

    def test_parse_gifti_nodes(self):
        """Test parsing node data."""
        jgii = gii2jgii(self.gii_path)

        nodes = get_node(jgii)
        self.assertIsNotNone(nodes)
        self.assertEqual(nodes.shape, (3, 3))
        np.testing.assert_almost_equal(nodes[0, 0], -16.072010, decimal=4)

    def test_parse_gifti_faces(self):
        """Test parsing face data with index conversion."""
        jgii = gii2jgii(self.gii_path)

        # Get 0-based faces
        faces = get_face(jgii, zero_based=True)
        self.assertIsNotNone(faces)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

        # Stored internally as 1-based
        stored = jgii["GIFTIData"]["MeshTri3"]["Data"]
        np.testing.assert_array_equal(stored[0], [1, 2, 3])

    def test_parse_gifti_coordsystem(self):
        """Test parsing coordinate system."""
        jgii = gii2jgii(self.gii_path)

        cs = jgii["GIFTIData"]["MeshVertex3"]["_DataInfo_"]["CoordSystem"]
        self.assertEqual(cs["DataSpace"], "talairach")
        self.assertEqual(len(cs["MatrixData"]), 4)


class TestRoundTrip(unittest.TestCase):
    """Tests for JGIFTI <-> GIFTI round-trip conversion."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir)

    def test_roundtrip_basic(self):
        """Test basic round-trip conversion."""
        node = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        face = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
            ],
            dtype=np.int32,
        )

        original = jgifticreate(node=node, face=face)

        # Save to GIFTI
        gii_path = os.path.join(self.tmp_dir, "test.gii")
        savegifti(original, gii_path, encoding="ASCII")

        # Load back
        loaded = loadgifti(gii_path)

        # Compare
        orig_nodes = get_node(original)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes, decimal=5)

        orig_faces = get_face(original)
        loaded_faces = get_face(loaded)
        np.testing.assert_array_equal(orig_faces, loaded_faces)

    def test_roundtrip_with_properties(self):
        """Test round-trip with node properties."""
        node = np.random.randn(10, 3).astype(np.float32)
        face = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        properties = {
            "Thickness": np.random.rand(10).astype(np.float32),
        }

        original = jgifticreate(node=node, face=face, properties=properties)

        gii_path = os.path.join(self.tmp_dir, "test_props.gii")
        savegifti(original, gii_path, encoding="GZipBase64Binary")

        loaded = loadgifti(gii_path)

        orig_thick = get_property(original, "Thickness")
        loaded_thick = get_property(loaded, "Thickness")

        # Property might be named differently after round-trip
        if loaded_thick is None:
            loaded_thick = get_property(loaded, "Shape")

        self.assertIsNotNone(loaded_thick)


class TestJGIFTIFileSaveLoad(unittest.TestCase):
    """Tests for JGIFTI file save/load operations."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir)

    def test_save_load_jgii(self):
        """Test saving and loading .jgii files."""
        try:
            import jdata
        except ImportError:
            self.skipTest("jdata module not available")

        node = np.random.randn(5, 3).astype(np.float32)
        original = jgifticreate(node=node)

        jgii_path = os.path.join(self.tmp_dir, "test.jgii")
        savejgifti(original, jgii_path)

        loaded = loadjgifti(jgii_path)

        orig_nodes = get_node(original)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes)

    def test_save_load_bgii(self):
        """Test saving and loading .bgii files."""
        try:
            import jdata
            import bjdata
        except ImportError:
            self.skipTest("jdata or bjdata module not available")

        node = np.random.randn(5, 3).astype(np.float32)
        original = jgifticreate(node=node)

        bgii_path = os.path.join(self.tmp_dir, "test.bgii")
        savejgifti(original, bgii_path)

        loaded = loadjgifti(bgii_path)

        orig_nodes = get_node(original)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes)


class TestEncodings(unittest.TestCase):
    """Tests for different data encodings."""

    def setUp(self):
        """Set up temporary directory and sample data."""
        self.tmp_dir = tempfile.mkdtemp()
        node = np.random.randn(100, 3).astype(np.float32)
        face = np.random.randint(0, 100, (50, 3)).astype(np.int32)
        self.sample_data = jgifticreate(node=node, face=face)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir)

    def test_ascii_encoding(self):
        """Test ASCII encoding."""
        path = os.path.join(self.tmp_dir, "ascii.gii")
        savegifti(self.sample_data, path, encoding="ASCII")

        loaded = loadgifti(path)
        orig_nodes = get_node(self.sample_data)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes, decimal=5)

    def test_base64_encoding(self):
        """Test Base64Binary encoding."""
        path = os.path.join(self.tmp_dir, "base64.gii")
        savegifti(self.sample_data, path, encoding="Base64Binary")

        loaded = loadgifti(path)
        orig_nodes = get_node(self.sample_data)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes)

    def test_gzip_encoding(self):
        """Test GZipBase64Binary encoding."""
        path = os.path.join(self.tmp_dir, "gzip.gii")
        savegifti(self.sample_data, path, encoding="GZipBase64Binary")

        loaded = loadgifti(path)
        orig_nodes = get_node(self.sample_data)
        loaded_nodes = get_node(loaded)
        np.testing.assert_array_almost_equal(orig_nodes, loaded_nodes)


class TestJGiftiClass(unittest.TestCase):
    """Tests for JGifti class."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir)

    def test_create_empty(self):
        """Test creating empty JGifti."""
        gii = JGifti()
        self.assertIsNotNone(gii.data)
        self.assertEqual(gii.version, "1.0")

    def test_create_with_data(self):
        """Test creating JGifti with node/face data."""
        node = np.random.randn(10, 3).astype(np.float32)
        face = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

        gii = JGifti(node=node, face=face)

        self.assertEqual(gii.nnode(), 10)
        self.assertEqual(gii.nface(), 2)
        np.testing.assert_array_equal(gii.node(), node)

    def test_create_with_anatomy(self):
        """Test creating JGifti with anatomy name."""
        node = np.random.randn(5, 3).astype(np.float32)

        gii = JGifti(node=node, anatomy="P001_L_pial")

        self.assertTrue(gii.is_multi_surface)
        self.assertIn("P001_L_pial", gii.surfaces)

    def test_property_access(self):
        """Test property get/set via dict-like interface."""
        node = np.random.randn(10, 3).astype(np.float32)
        gii = JGifti(node=node)

        thickness = np.random.rand(10).astype(np.float32)
        gii["Thickness"] = thickness

        self.assertIn("Thickness", gii)
        np.testing.assert_array_equal(gii["Thickness"], thickness)

    def test_property_names(self):
        """Test getting property names."""
        node = np.random.randn(5, 3).astype(np.float32)

        gii = JGifti(node=node)
        gii["Thickness"] = np.array([1, 2, 3, 4, 5])
        gii["Curvature"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        names = gii.property_names()
        self.assertIn("Thickness", names)
        self.assertIn("Curvature", names)

    def test_labels(self):
        """Test label table access."""
        gii = JGifti()

        gii.add_label(0, "Unknown", [0.5, 0.5, 0.5, 1.0])
        gii.add_label(1, "V1", [1.0, 0.0, 0.0, 1.0])

        self.assertEqual(gii.get_label_name(0), "Unknown")
        self.assertEqual(gii.get_label_name(1), "V1")
        self.assertEqual(gii.get_label_color(1), [1.0, 0.0, 0.0, 1.0])

    def test_coord_system(self):
        """Test coordinate system access."""
        node = np.random.randn(5, 3).astype(np.float32)
        gii = JGifti(node=node)

        gii.set_coord_system("scanner_anat", "mni_152")

        self.assertEqual(gii.data_space(), "scanner_anat")
        self.assertEqual(gii.transformed_space(), "mni_152")

    def test_metadata(self):
        """Test metadata access."""
        gii = JGifti()

        gii.set_metadata("SubjectID", "sub001")
        gii.set_metadata("Date", "2024-01-15")

        meta = gii.metadata
        self.assertEqual(meta["SubjectID"], "sub001")
        self.assertEqual(meta["Date"], "2024-01-15")

    def test_save_load_roundtrip(self):
        """Test JGifti save/load round-trip."""
        node = np.random.randn(10, 3).astype(np.float32)
        face = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

        gii = JGifti(node=node, face=face)
        gii["Thickness"] = np.random.rand(10).astype(np.float32)

        # Save and load
        path = os.path.join(self.tmp_dir, "test.gii")
        gii.save(path, encoding="ASCII")

        loaded = JGifti(path)

        np.testing.assert_array_almost_equal(gii.node(), loaded.node(), decimal=5)
        np.testing.assert_array_equal(gii.face(), loaded.face())

    def test_add_surface(self):
        """Test adding surfaces to create multi-surface structure."""
        gii = JGifti()

        pial_nodes = np.random.randn(5, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

        gii.add_surface("P001_L_pial", node=pial_nodes, face=faces)

        white_nodes = np.random.randn(5, 3).astype(np.float32)
        gii.add_surface(
            "P001_L_white", node=white_nodes, share_topology_with="P001_L_pial"
        )

        self.assertTrue(gii.is_multi_surface)
        self.assertEqual(len(gii.surfaces), 2)
        self.assertIn("P001_L_pial", gii.surfaces)
        self.assertIn("P001_L_white", gii.surfaces)

    def test_default_anatomy(self):
        """Test default anatomy setting."""
        gii = JGifti()

        gii.add_surface("P001_L_pial", node=np.random.randn(5, 3).astype(np.float32))
        gii.add_surface("P001_L_white", node=np.random.randn(5, 3).astype(np.float32))

        # Set default anatomy
        gii.anatomy = "P001_L_pial"

        # Access without specifying anatomy uses default
        nodes = gii.node()
        pial_nodes = gii.node("P001_L_pial")

        np.testing.assert_array_equal(nodes, pial_nodes)

    def test_copy(self):
        """Test deep copy."""
        node = np.random.randn(5, 3).astype(np.float32)
        gii = JGifti(node=node)
        gii["Thickness"] = np.array([1, 2, 3, 4, 5])

        copied = gii.copy()

        # Modify original
        gii["Thickness"] = np.array([10, 20, 30, 40, 50])

        # Copy should be unchanged
        np.testing.assert_array_equal(copied["Thickness"], [1, 2, 3, 4, 5])

    def test_repr_str(self):
        """Test string representations."""
        node = np.random.randn(10, 3).astype(np.float32)
        face = np.array([[0, 1, 2]], dtype=np.int32)

        gii = JGifti(node=node, face=face)
        gii["Thickness"] = np.random.rand(10)

        repr_str = repr(gii)
        self.assertIn("JGifti", repr_str)
        self.assertIn("node=10", repr_str)
        self.assertIn("face=1", repr_str)

        str_str = str(gii)
        self.assertIn("JGifti Surface Mesh", str_str)
        self.assertIn("Nodes: 10", str_str)


class TestJGiftiMultiSurface(unittest.TestCase):
    """Tests for JGifti multi-surface functionality."""

    def setUp(self):
        """Create multi-surface JGifti."""
        self.gii = JGifti()

        pial = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        white = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        self.gii.add_surface(
            "pial",
            node=pial,
            face=faces,
            properties={"Thickness": np.array([2.5, 2.3, 2.8])},
        )
        self.gii.add_surface("white", node=white, share_topology_with="pial")

    def test_surfaces_list(self):
        """Test getting surfaces list."""
        surfaces = self.gii.surfaces
        self.assertIn("pial", surfaces)
        self.assertIn("white", surfaces)

    def test_node_by_anatomy(self):
        """Test getting nodes by anatomy."""
        pial_nodes = self.gii.node("pial")
        white_nodes = self.gii.node("white")

        self.assertEqual(pial_nodes[0, 2], 0)
        self.assertEqual(white_nodes[0, 2], -1)

    def test_property_by_anatomy(self):
        """Test getting property by anatomy."""
        thickness = self.gii.get_property("Thickness", "pial")
        np.testing.assert_array_almost_equal(thickness, [2.5, 2.3, 2.8])

    def test_shared_topology(self):
        """Test shared topology via _DataLink_."""
        pial_faces = self.gii.face("pial")
        white_faces = self.gii.face("white")

        np.testing.assert_array_equal(pial_faces, white_faces)

    def test_nnode_nface_by_anatomy(self):
        """Test nnode/nface with anatomy."""
        self.assertEqual(self.gii.nnode("pial"), 3)
        self.assertEqual(self.gii.nnode("white"), 3)
        self.assertEqual(self.gii.nface("pial"), 1)

    def test_str_multi_surface(self):
        """Test string representation for multi-surface."""
        str_repr = str(self.gii)
        self.assertIn("Surfaces:", str_repr)
        self.assertIn("pial", str_repr)
        self.assertIn("white", str_repr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
