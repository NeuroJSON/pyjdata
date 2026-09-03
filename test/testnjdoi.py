"""Tests for jdata.njdoi -- DataCite metadata for a dataset version.

A DOI is a permanent promise, so the record has to be right in two ways: it must
carry the five mandatory DataCite properties whatever the source dataset omits,
and it must carry the content fingerprint, which is what lets someone verify
that the bytes they downloaded are the bytes the DOI was minted for.
"""

import os
import sys
import json
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jdata.njdoi import datacite, creators_from_authors, landing_url

FULL_DOC = {
    ".neurojson": {
        "Version": "3.1.0",
        "SourceCommit": "d" * 40,
        "Fingerprint": "e" * 64,
        "Files": 133,
        "Bytes": 2416199965,
    },
    "dataset_description.json": {
        "Name": "Delayed Free Recall of Word Lists",
        "BIDSVersion": "1.7.0",
        "Authors": ["Herrema, Haydn G.", "Michael J. Kahana 0000-0001-8122-9525"],
        "License": "CC0",
        "Funding": ["NIH: R01-NS-106611", "DARPA RAM: N66001-14-2-4032"],
        "DatasetDOI": "doi:10.18112/openneuro.ds004789.v3.1.0",
        "ReferencesAndLinks": ["10.1038/s41467-000", "https://example.org/paper"],
    },
    "README": "A study of delayed free recall.",
    "sub-01": {"anat": {"sub-01_T1w.nii.gz": {}}, "func": {"sub-01_task-recall_bold.nii.gz": {}}},
    "sub-02": {"ses-01": {"eeg": {"sub-02_ses-01_task-recall_eeg.edf": {}}}},
}

MANIFEST = [
    {"path": "sub-01/anat/sub-01_T1w.nii.gz", "sha256": "a" * 64, "size": 1},
    {"path": "sub-01/func/sub-01_task-recall_bold.nii.gz", "sha256": "b" * 64, "size": 2},
    {"path": "participants.tsv", "sha256": "c" * 64, "size": 3},
    {"path": "sub-02/ses-01/eeg/sub-02_ses-01_task-recall_eeg.edf", "sha256": "d" * 64, "size": 4},
]


class TestCreators(unittest.TestCase):
    def test_comma_form_is_split_into_family_and_given(self):
        creator = creators_from_authors(["Herrema, Haydn G."])[0]
        self.assertEqual(creator["familyName"], "Herrema")
        self.assertEqual(creator["givenName"], "Haydn G.")
        self.assertEqual(creator["nameType"], "Personal")

    def test_plain_form_takes_the_last_token_as_family_name(self):
        creator = creators_from_authors(["Michael J. Kahana"])[0]
        self.assertEqual(creator["familyName"], "Kahana")
        self.assertEqual(creator["givenName"], "Michael J.")

    def test_embedded_orcid_is_lifted_out_of_the_name(self):
        creator = creators_from_authors(["Jane Doe 0000-0001-8122-9525"])[0]
        self.assertNotIn("0000-0001", creator["name"])
        self.assertEqual(
            creator["nameIdentifiers"][0]["nameIdentifier"],
            "https://orcid.org/0000-0001-8122-9525",
        )
        self.assertEqual(creator["nameIdentifiers"][0]["nameIdentifierScheme"], "ORCID")

    def test_orcid_with_trailing_x_checksum(self):
        creator = creators_from_authors(["A B 0000-0002-1825-009X"])[0]
        self.assertIn("0000-0002-1825-009X", creator["nameIdentifiers"][0]["nameIdentifier"])

    def test_missing_authors_still_yields_a_creator(self):
        """DataCite requires at least one creator, so a fallback is mandatory."""
        for value in ([], None, [""]):
            creators = creators_from_authors(value)
            self.assertEqual(len(creators), 1)
            self.assertEqual(creators[0]["nameType"], "Organizational")

    def test_single_token_name_has_no_split(self):
        creator = creators_from_authors(["Anonymous"])[0]
        self.assertNotIn("familyName", creator)


class TestLandingUrl(unittest.TestCase):
    def test_version_is_included(self):
        self.assertEqual(
            landing_url("db", "ds1", "1.0.0"), "https://neurojson.io/db/db/ds1?ver=1.0.0"
        )

    def test_version_omitted_when_unknown(self):
        self.assertEqual(landing_url("db", "ds1"), "https://neurojson.io/db/db/ds1")

    def test_base_is_configurable(self):
        self.assertTrue(landing_url("d", "s", base="https://x/y/").startswith("https://x/y/d/s"))


class TestDataciteRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = datacite(
            FULL_DOC, "openneuro_full", "ds004789", manifest=MANIFEST, publication_year=2026
        )

    def test_mandatory_properties_present(self):
        for key in ("creators", "titles", "publisher", "publicationYear", "types"):
            self.assertIn(key, self.record)
        self.assertEqual(self.record["types"]["resourceTypeGeneral"], "Dataset")
        self.assertEqual(self.record["publicationYear"], 2026)

    def test_title_and_version(self):
        self.assertEqual(self.record["titles"][0]["title"], "Delayed Free Recall of Word Lists")
        self.assertEqual(self.record["version"], "3.1.0")

    def test_landing_url_pins_the_version(self):
        self.assertEqual(
            self.record["url"], "https://neurojson.io/db/openneuro_full/ds004789?ver=3.1.0"
        )

    def test_fingerprint_and_commit_are_alternate_identifiers(self):
        """This is what makes the DOI independently verifiable."""
        kinds = {
            a["alternateIdentifierType"]: a["alternateIdentifier"]
            for a in self.record["alternateIdentifiers"]
        }
        self.assertEqual(kinds["NeuroJSON-Fingerprint"], "sha256:" + "e" * 64)
        self.assertEqual(kinds["Git-Commit"], "d" * 40)
        self.assertEqual(kinds["Accession"], "ds004789")

    def test_upstream_doi_becomes_a_variant_relation(self):
        rels = {
            r["relatedIdentifier"]: r["relationType"] for r in self.record["relatedIdentifiers"]
        }
        self.assertEqual(rels["10.18112/openneuro.ds004789.v3.1.0"], "IsVariantFormOf")

    def test_references_are_classified_as_doi_or_url(self):
        by_id = {r["relatedIdentifier"]: r for r in self.record["relatedIdentifiers"]}
        self.assertEqual(by_id["10.1038/s41467-000"]["relatedIdentifierType"], "DOI")
        self.assertEqual(by_id["https://example.org/paper"]["relatedIdentifierType"], "URL")

    def test_license_is_mapped_to_a_rights_uri(self):
        rights = self.record["rightsList"][0]
        self.assertEqual(rights["rightsIdentifier"], "CC0")
        self.assertTrue(rights["rightsUri"].startswith("https://creativecommons.org/"))

    def test_funding_is_split_into_funder_and_award(self):
        funders = {f["funderName"]: f.get("awardNumber") for f in self.record["fundingReferences"]}
        self.assertEqual(funders["NIH"], "R01-NS-106611")
        self.assertEqual(funders["DARPA RAM"], "N66001-14-2-4032")

    def test_sizes_come_from_the_metadata_block(self):
        self.assertIn("2416199965 bytes", self.record["sizes"])
        self.assertIn("133 files", self.record["sizes"])

    def test_formats_derive_from_the_manifest_with_compound_extensions(self):
        self.assertEqual(self.record["formats"], [".edf", ".nii.gz", ".tsv"])

    def test_subjects_list_modalities_and_tasks(self):
        subjects = {s["subject"] for s in self.record["subjects"]}
        self.assertTrue({"anat", "func", "eeg"}.issubset(subjects))
        self.assertIn("task-recall", subjects)

    def test_readme_becomes_the_abstract(self):
        self.assertEqual(self.record["descriptions"][0]["descriptionType"], "Abstract")
        self.assertIn("delayed free recall", self.record["descriptions"][0]["description"])

    def test_record_is_json_serialisable(self):
        json.dumps(self.record)


class TestDataciteDegradedInput(unittest.TestCase):
    """A DOI must still be mintable for a dataset with almost no metadata."""

    def test_bare_document(self):
        record = datacite({}, "db", "ds999")
        self.assertEqual(record["titles"][0]["title"], "ds999")
        self.assertEqual(len(record["creators"]), 1)
        self.assertIn("publicationYear", record)
        self.assertNotIn("version", record)
        json.dumps(record)

    def test_description_of_the_wrong_type_is_ignored(self):
        record = datacite({"dataset_description.json": "not a dict"}, "db", "ds1")
        self.assertEqual(record["titles"][0]["title"], "ds1")

    def test_readme_variants_are_found(self):
        for key in ("README.md", "README.rst"):
            record = datacite({key: "text body"}, "db", "ds1")
            self.assertIn("text body", record["descriptions"][0]["description"])

    def test_long_readme_is_truncated(self):
        record = datacite({"README": "x" * 9000}, "db", "ds1")
        self.assertLessEqual(len(record["descriptions"][0]["description"]), 4000)

    def test_explicit_doi_is_echoed_as_an_identifier(self):
        record = datacite({}, "db", "ds1", doi="10.5072/neurojson.ds1.v1")
        self.assertEqual(record["doi"], "10.5072/neurojson.ds1.v1")
        self.assertEqual(record["identifiers"][0]["identifierType"], "DOI")

    def test_unknown_license_is_kept_verbatim(self):
        record = datacite({"dataset_description.json": {"License": "Custom terms"}}, "db", "ds1")
        self.assertEqual(record["rightsList"], [{"rights": "Custom terms"}])


class TestRealConvertedDocument(unittest.TestCase):
    """Run against a real converted document if one is available."""

    PATH = "/lake/neurojson/prep/openneuro_full/json2/ds000001/1.0.0/doc.json"

    @unittest.skipUnless(os.path.isfile(PATH), "no converted output available")
    def test_ds000001(self):
        with open(self.PATH, encoding="utf-8") as fid:
            doc = json.load(fid)
        record = datacite(doc, "openneuro_full", "ds000001")
        self.assertEqual(record["titles"][0]["title"], "Balloon Analog Risk-taking Task")
        self.assertEqual(record["version"], "1.0.0")
        self.assertEqual(len(record["creators"]), 4)
        kinds = {a["alternateIdentifierType"] for a in record["alternateIdentifiers"]}
        self.assertIn("NeuroJSON-Fingerprint", kinds)
        json.dumps(record)


if __name__ == "__main__":
    unittest.main()
