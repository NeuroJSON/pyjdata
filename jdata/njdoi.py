"""
DataCite metadata for a converted dataset version.

A DOI is only useful if the thing it points at cannot change underneath it.
:mod:`jdata.njbids` provides that half: a dataset version converts to a
byte-reproducible document with a content fingerprint.  This module provides the
other half -- the metadata record a registrar needs, derived from the digest so
that it is generated rather than hand-maintained.

The output follows the DataCite Metadata Schema 4.x JSON form.  The five
mandatory properties (``creators``, ``titles``, ``publisher``,
``publicationYear``, ``types``) are always emitted; the rest are filled in from
whatever ``dataset_description.json`` provides.

Two fields deserve note:

``version``
    the derived dataset version, so each version gets its own DOI.

``alternateIdentifiers``
    carries the NeuroJSON content fingerprint.  That is what lets a third party
    verify that the bytes they downloaded are the bytes the DOI was minted for,
    without trusting the server -- recompute the manifest and compare.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import re
import json
import datetime

__all__ = ["datacite", "creators_from_authors", "landing_url"]

_DEFAULT_PUBLISHER = "NeuroJSON"
_ORCID = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")


def creators_from_authors(authors):
    """Map ``dataset_description.json:Authors`` onto DataCite ``creators``.

    BIDS gives free-text names, so the split is heuristic: a name containing a
    comma is treated as "Family, Given", otherwise the last whitespace-separated
    token is taken as the family name.  An embedded ORCID is extracted rather
    than left inside the name string.
    """
    out = []
    for entry in authors or []:
        name = str(entry).strip()
        if not name:
            continue
        orcid = None
        match = _ORCID.search(name)
        if match:
            orcid = match.group(1)
            name = name.replace(match.group(0), "").strip(" ,;()")
        creator = {"name": name, "nameType": "Personal"}
        if "," in name:
            family, _, given = name.partition(",")
            creator["familyName"] = family.strip()
            creator["givenName"] = given.strip()
        else:
            parts = name.split()
            if len(parts) > 1:
                creator["familyName"] = parts[-1]
                creator["givenName"] = " ".join(parts[:-1])
        if orcid:
            creator["nameIdentifiers"] = [
                {
                    "nameIdentifier": "https://orcid.org/%s" % orcid,
                    "nameIdentifierScheme": "ORCID",
                    "schemeUri": "https://orcid.org",
                }
            ]
        out.append(creator)
    return out or [{"name": _DEFAULT_PUBLISHER, "nameType": "Organizational"}]


def landing_url(db, ds, version=None, base="https://neurojson.io/db"):
    """Landing page a DOI should resolve to."""
    url = "%s/%s/%s" % (base.rstrip("/"), db, ds)
    return "%s?ver=%s" % (url, version) if version else url


def _license_rights(value):
    known = {
        "CC0": ("CC0 1.0 Universal", "https://creativecommons.org/publicdomain/zero/1.0/"),
        "CC0-1.0": ("CC0 1.0 Universal", "https://creativecommons.org/publicdomain/zero/1.0/"),
        "CC-BY-4.0": (
            "Creative Commons Attribution 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        ),
        "CC-BY": (
            "Creative Commons Attribution 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        ),
        "PDDL": ("Open Data Commons PDDL", "https://opendatacommons.org/licenses/pddl/"),
    }
    key = str(value).strip()
    if key in known:
        title, uri = known[key]
        return [{"rights": title, "rightsUri": uri, "rightsIdentifier": key}]
    return [{"rights": key}] if key else []


def _formats(manifest):
    """Distinct file extensions, as DataCite ``formats``."""
    seen = []
    for entry in manifest or []:
        path = entry.get("path", "")
        lower = path.lower()
        ext = ""
        for compound in (".nii.gz", ".tsv.gz", ".csv.gz", ".gii.gz"):
            if lower.endswith(compound):
                ext = compound
                break
        if not ext:
            ext = os.path.splitext(lower)[1]
        if ext and ext not in seen:
            seen.append(ext)
    return sorted(seen)


def _related(description, doc_id=None):
    related = []
    doi = str(description.get("DatasetDOI") or "").strip()
    if doi:
        related.append(
            {
                "relatedIdentifier": doi.replace("doi:", ""),
                "relatedIdentifierType": "DOI",
                "relationType": "IsVariantFormOf",
            }
        )
    for link in description.get("ReferencesAndLinks") or []:
        text = str(link).strip()
        if not text:
            continue
        if re.match(r"^10\.\d{4,9}/", text) or text.lower().startswith("doi:"):
            related.append(
                {
                    "relatedIdentifier": text.replace("doi:", ""),
                    "relatedIdentifierType": "DOI",
                    "relationType": "IsSupplementTo",
                }
            )
        elif text.lower().startswith("http"):
            related.append(
                {
                    "relatedIdentifier": text,
                    "relatedIdentifierType": "URL",
                    "relationType": "IsSupplementTo",
                }
            )
    return related


def _subjects(doc):
    """Modalities and task names, as DataCite ``subjects``.

    Task labels live in BIDS *filenames* nested under each subject, not in the
    top-level keys, so the subject subtrees have to be walked.
    """
    modalities = []
    tasks = []

    def visit(node, depth):
        if depth > 4 or not isinstance(node, dict):
            return
        for key, value in node.items():
            match = re.search(r"task-([A-Za-z0-9]+)", key)
            if match and match.group(1) not in tasks:
                tasks.append(match.group(1))
            if "." in key:
                continue
            if key.startswith("ses-"):
                visit(value, depth + 1)
                continue
            if depth >= 1 and key not in modalities:
                modalities.append(key)
            visit(value, depth + 1)

    for key, value in doc.items():
        if key.lower().startswith("sub-") and isinstance(value, dict):
            visit(value, 1)

    return [{"subject": s} for s in sorted(modalities)] + [
        {"subject": "task-%s" % t} for t in sorted(tasks)
    ]


def datacite(
    doc,
    db,
    ds,
    manifest=None,
    publisher=_DEFAULT_PUBLISHER,
    publication_year=None,
    landing_base="https://neurojson.io/db",
    doi=None,
    resource_type="Dataset",
):
    """Build a DataCite 4.x metadata record for one converted dataset version.

    Parameters
    ----------
    doc : dict
        A converted document, including its ``.neurojson`` block.
    db, ds : str
        Database and dataset identifiers, used for the landing page URL.
    manifest : list, optional
        Manifest entries, used to derive ``formats`` and ``sizes``.
    publication_year : int, optional
        Defaults to the current year.  This is metadata about the *publication*
        rather than the dataset content, so it is intentionally not part of the
        fingerprinted payload.

    Returns
    -------
    dict
        The record, ready to submit or to write alongside the version archive.
    """
    meta = doc.get(".neurojson", {}) or {}
    description = doc.get("dataset_description.json") or {}
    if not isinstance(description, dict):
        description = {}
    version = meta.get("Version")
    name = description.get("Name") or ds

    readme = ""
    for key in ("README", "README.md", "README.rst"):
        if isinstance(doc.get(key), str):
            readme = doc[key]
            break

    record = {
        "creators": creators_from_authors(description.get("Authors")),
        "titles": [{"title": str(name).strip()}],
        "publisher": publisher,
        "publicationYear": int(publication_year or datetime.date.today().year),
        "types": {"resourceTypeGeneral": resource_type, "resourceType": "BIDS dataset"},
        "url": landing_url(db, ds, version, base=landing_base),
        "schemaVersion": "http://datacite.org/schema/kernel-4",
    }
    if doi:
        record["doi"] = doi
        record["identifiers"] = [{"identifier": doi, "identifierType": "DOI"}]
    if version:
        record["version"] = version

    alternates = [{"alternateIdentifier": ds, "alternateIdentifierType": "Accession"}]
    if meta.get("Fingerprint"):
        alternates.append(
            {
                "alternateIdentifier": "sha256:%s" % meta["Fingerprint"],
                "alternateIdentifierType": "NeuroJSON-Fingerprint",
            }
        )
    if meta.get("SourceCommit"):
        alternates.append(
            {
                "alternateIdentifier": meta["SourceCommit"],
                "alternateIdentifierType": "Git-Commit",
            }
        )
    record["alternateIdentifiers"] = alternates

    if readme:
        record["descriptions"] = [
            {
                "description": readme.strip()[:4000],
                "descriptionType": "Abstract",
                "lang": "en",
            }
        ]
    rights = _license_rights(description.get("License"))
    if rights:
        record["rightsList"] = rights
    related = _related(description)
    if related:
        record["relatedIdentifiers"] = related
    subjects = _subjects(doc)
    if subjects:
        record["subjects"] = subjects

    funding = []
    for grant in description.get("Funding") or []:
        text = str(grant).strip()
        if not text:
            continue
        funder, _, award = text.partition(":")
        entry = {"funderName": funder.strip() or text}
        if award.strip():
            entry["awardNumber"] = award.strip()
        funding.append(entry)
    if funding:
        record["fundingReferences"] = funding

    sizes = []
    if meta.get("Bytes"):
        sizes.append("%d bytes" % int(meta["Bytes"]))
    if meta.get("Files"):
        sizes.append("%d files" % int(meta["Files"]))
    if sizes:
        record["sizes"] = sizes
    formats = _formats(manifest)
    if formats:
        record["formats"] = formats

    contributors = []
    for entry in (
        description.get("Acknowledgements", [])
        if isinstance(description.get("Acknowledgements"), list)
        else []
    ):
        contributors.append({"name": str(entry), "contributorType": "Other"})
    if contributors:
        record["contributors"] = contributors

    return record
