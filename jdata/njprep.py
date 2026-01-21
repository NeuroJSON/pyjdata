"""
Folder-to-JSON conversion functions for neurojson.py

Converts neuroimaging dataset folders to JSON format, separating
human-readable metadata from binary data. Mirrors functionality of
neuroj/njprep shell scripts.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import re
import json
import hashlib
import glob
import shutil
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Import jdata functions from the same package
from .jfile import savejson, saveb, loadt, savejd, load, loadsnirf
from .jnifti import nii2jnii
from .csv import load_csv_tsv, is_enum_encoded, decode_enum_column, json2tsv

__all__ = [
    "dataset2json",
    "NJPREP_DEFAULT",
]

# Default configuration (can be overridden via environment or kwargs)
NJPREP_DEFAULT = {
    "MAX_TSV": 65536,
    "MAX_JSON": 65536,
    "MAX_BJSON": 65536,
    "MAX_BVEC": 65536,
    "MAX_MAT": 2000000,
    "FALLBACK_URL": "",
    "ATTACH_URL_TEMPLATE": "https://neurojson.org/io/stat.cgi?action=get&db={db}&doc={ds}&file={file}",
}


def dataset2json(
    inputroot: str,
    outputroot: str,
    dbname: str = None,
    dsname: str = None,
    filename: str = None,
    convert: bool = False,
    threads: int = 4,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convert neuroimaging dataset folders to JSON format.

    This function mirrors the functionality of neuroj/njprep shell scripts,
    converting various neuroimaging file formats to JSON while separating
    human-readable metadata from binary data.

    Args:
        inputroot: Path to the input root folder containing datasets
        outputroot: Path to the output folder for JSON files
        dbname: Database name (e.g., 'openneuro'). If None, uses last segment of inputroot
        dsname: Dataset name. If None or '*', processes all datasets
        filename: Specific file to convert (optional)
        convert: If True, actually perform conversion. If False, dry-run (preview commands)
        threads: Number of parallel threads for file conversion (default: 4)
        **kwargs: Additional options:
            - max_tsv: Max TSV file size to inline (default: 65536)
            - max_json: Max JSON file size to inline (default: 65536)
            - max_bjson: Max binary JSON file size to inline (default: 65536)
            - max_bvec: Max bvec/bval file size to inline (default: 65536)
            - max_mat: Max MAT file size to inline (default: 2000000)
            - fallback_url: URL template for external data links
            - attach_url: URL template for attachments

    Returns:
        dict: Conversion results including:
            - status: 'success', 'partial', or 'error'
            - files_processed: Number of files processed
            - datasets_processed: List of dataset names processed
            - errors: List of any errors encountered
            - commands: List of commands (in dry-run mode)

    Example:
        # Dry-run to preview commands
        result = folder2json('/data/openneuro', '/output/json', 'openneuro', convert=False)

        # Convert a single dataset with 8 threads
        result = dataset2json('/data/openneuro', '/output/json', 'openneuro', 'ds000001', convert=True, threads=8)

        # Convert all datasets with 8 threads (files across all datasets in parallel)
        result = dataset2json('/data/openneuro', '/output/json', 'openneuro', '*', convert=True, threads=8)
    """
    # Build configuration from defaults, environment variables, and kwargs
    config = _build_config(kwargs)

    # Normalize paths
    inputroot = inputroot.rstrip("/\\")
    outputroot = outputroot.rstrip("/\\")
    dbname = dbname or os.path.basename(inputroot)

    result = {
        "status": "success",
        "files_processed": 0,
        "datasets_processed": [],
        "errors": [],
        "commands": [],
    }

    # Case 1: Convert a single file
    if filename:
        if convert:
            try:
                _convert_file(inputroot, outputroot, dbname, dsname, filename, config)
                result["files_processed"] = 1
            except Exception as e:
                result["errors"].append(f"{filename}: {e}")
        else:
            result["commands"].append(f"convert_file({dbname}, {dsname}, {filename})")
        return _finalize_result(result)

    # Case 2: Convert dataset(s)
    datasets = _get_dataset_list(inputroot, dsname)

    # Dry-run mode: just list commands
    if not convert:
        for ds in datasets:
            result["commands"].append(
                f"convert_dataset({dbname}, {ds}, threads={threads})"
            )
            result["commands"].append(f"mergejson({outputroot}/{ds})")
            result["commands"].append(f"bids2json({outputroot}, {ds})")
        return result

    # Collect all files from all datasets
    all_files = []
    for ds in datasets:
        if os.path.isfile(f"{outputroot}/{ds}.json"):
            print(f"Dataset {ds} already converted, skipping")
            continue

        # Clear hash directory
        hashdir = os.path.join(outputroot, ".hash", ds)
        if os.path.exists(hashdir):
            shutil.rmtree(hashdir)

        # Collect files from this dataset
        dspath = os.path.join(inputroot, ds)
        for root, dirs, files in os.walk(dspath):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                all_files.append((ds, os.path.join(root, f)))
        result["datasets_processed"].append(ds)

    print(
        f"Converting {len(all_files)} files across {len(result['datasets_processed'])} datasets with {threads} threads"
    )

    # Convert files in parallel
    if threads > 1 and len(all_files) > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(
                    _convert_file_safe, inputroot, outputroot, dbname, ds, fp, config
                ): (ds, fp)
                for ds, fp in all_files
            }
            for future in as_completed(futures):
                success, error = future.result()
                if success:
                    result["files_processed"] += 1
                else:
                    result["errors"].append(f"{futures[future][1]}: {error}")
    else:
        for ds, fp in all_files:
            success, error = _convert_file_safe(
                inputroot, outputroot, dbname, ds, fp, config
            )
            if success:
                result["files_processed"] += 1
            else:
                result["errors"].append(f"{fp}: {error}")

    print(f"Converted {result['files_processed']}/{len(all_files)} files")

    # Post-processing: merge JSON files for each dataset
    for ds in result["datasets_processed"]:
        outpath = os.path.join(outputroot, ds)
        # Copy dataset_description.json if needed
        dd_in, dd_out = os.path.join(
            inputroot, ds, "dataset_description.json"
        ), os.path.join(outpath, "dataset_description.json")
        if os.path.exists(dd_in) and not os.path.exists(dd_out):
            os.makedirs(outpath, exist_ok=True)
            shutil.copy2(dd_in, dd_out)
        if os.path.isdir(outpath):
            _mergejson(outpath)
            _bids2json(outputroot, ds)

    return _finalize_result(result)


# =============================================================================
# Configuration and utility helpers
# =============================================================================


def _build_config(kwargs: Dict) -> Dict:
    """Build configuration from defaults, environment variables, and kwargs."""
    config = NJPREP_DEFAULT.copy()
    for key in ["max_tsv", "max_json", "max_bjson", "max_bvec", "max_mat"]:
        env_val = os.environ.get(f"NJPREP_{key.upper()}")
        if env_val:
            config[key.upper()] = int(env_val)
        if key in kwargs:
            config[key.upper()] = kwargs[key]
    config["FALLBACK_URL"] = kwargs.get(
        "fallback_url", os.environ.get("NJPREP_FALLBACK_URL", "")
    )
    if "attach_url" in kwargs:
        config["ATTACH_URL_TEMPLATE"] = kwargs["attach_url"]
    return config


def _get_dataset_list(inputroot: str, dsname: str) -> List[str]:
    """Get list of dataset names to process."""
    if dsname and dsname != "*" and not re.search(r"\s+", dsname):
        return [dsname]
    if dsname and re.search(r"\s+", dsname):
        return dsname.split()
    return [
        os.path.basename(d.rstrip("/\\"))
        for d in glob.glob(f"{inputroot}/*/")
        if os.path.isdir(d)
    ]


def _finalize_result(result: Dict) -> Dict:
    """Set final status based on errors."""
    if result["errors"]:
        result["status"] = (
            "partial"
            if result["datasets_processed"] or result["files_processed"]
            else "error"
        )
    return result


def _convert_file_safe(
    inputroot: str,
    outputroot: str,
    dbname: str,
    dsname: str,
    filepath: str,
    config: Dict,
) -> tuple:
    """Thread-safe wrapper for _convert_file. Returns (success, error_message)."""
    try:
        _convert_file(inputroot, outputroot, dbname, dsname, filepath, config)
        return True, None
    except Exception as e:
        return False, str(e)


# =============================================================================
# Hash and path utilities
# =============================================================================


def _file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Calculate hash of file contents."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_duplicate(
    outputroot: str, dsname: str, contenthash: str, suffix: str
) -> str:
    """Check if content hash already exists, return JSONPath reference or None."""
    hashpathfile = os.path.join(
        outputroot, ".hash", dsname, f"{contenthash}{suffix}.path"
    )
    if os.path.exists(hashpathfile):
        with open(hashpathfile, "r") as f:
            original_path = f.read().strip()
        return "$." + original_path.replace(".", "\\.").replace("/", ".")
    return None


def _register_hash(
    outputroot: str, dsname: str, contenthash: str, suffix: str, relpath: str
) -> None:
    """Register content hash to path mapping for deduplication."""
    hashdir = os.path.join(outputroot, ".hash", dsname)
    os.makedirs(hashdir, exist_ok=True)
    hashpathfile = os.path.join(hashdir, f"{contenthash}{suffix}.path")
    if not os.path.exists(hashpathfile):
        with open(hashpathfile, "w") as f:
            f.write(relpath)


def _save_json(filepath: str, data: Any) -> None:
    """Save data as JSON file, converting numpy types."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, separators=(",", ":"))


def _save_as_tsv(filepath: str, data: Dict) -> None:
    """Save JSON data as TSV file, decoding enum-encoded columns if present."""
    json2tsv(data, filepath, delimiter="\t")


def _save_as_csv(filepath: str, data: Dict) -> None:
    """Save JSON data as CSV file, decoding enum-encoded columns if present."""
    json2tsv(data, filepath, delimiter=",")


def _to_serializable(data: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {k: _to_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_serializable(item) for item in data]
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    if isinstance(data, np.bool_):
        return bool(data)
    return data


# =============================================================================
# File extension utilities
# =============================================================================


def _get_extension(filepath: str) -> str:
    """Get file extension, handling compound extensions like .nii.gz."""
    filepath_lower = filepath.lower()
    for ext in [".nii.gz", ".tsv.gz", ".csv.gz", ".img.gz"]:
        if filepath_lower.endswith(ext):
            return ext
    return os.path.splitext(filepath_lower)[1]


def _has_ext(filepath: str, extensions: List[str]) -> bool:
    """Check if file has one of the given extensions (case-insensitive)."""
    filepath_lower = filepath.lower()
    for ext in extensions:
        if not ext.startswith("."):
            ext = "." + ext
        if filepath_lower.endswith(ext.lower()):
            return True
    return False


def _is_text_file(fname: str) -> bool:
    """Check if filename indicates a known text file."""
    return os.path.splitext(fname)[0].upper() in [
        "README",
        "CHANGES",
        "CITATION",
        "LICENSE",
    ]


# =============================================================================
# Main file conversion
# =============================================================================


def _convert_file(
    inputroot: str,
    outputroot: str,
    dbname: str,
    dsname: str,
    filepath: str,
    config: Dict,
) -> None:
    """Convert a single file to JSON format."""
    dsname = dsname.rstrip("/\\")
    dspath = os.path.join(inputroot, dsname)
    fname = os.path.basename(filepath)
    fdir = os.path.dirname(filepath)

    # Calculate paths
    relpath = os.path.relpath(filepath, inputroot)
    relpath2 = os.path.relpath(filepath, dspath)
    if relpath2.startswith("./"):
        relpath2 = relpath2[2:]

    outputdir = os.path.dirname(os.path.join(outputroot, relpath))
    os.makedirs(outputdir, exist_ok=True)

    # Get file info
    is_link = os.path.islink(filepath)
    filesize = (
        os.path.getsize(filepath) if os.path.isfile(filepath) and not is_link else 0
    )
    contenthash = _file_hash(filepath) if filesize > 0 else None

    attach_url = config["ATTACH_URL_TEMPLATE"].format(db=dbname, ds=dsname, file="")
    fallback_url = config["FALLBACK_URL"]
    outfile = os.path.join(outputroot, relpath)

    print(f"Converting: {filepath}")

    # Directory - skip
    if os.path.isdir(filepath):
        return

    # Broken symlink
    if is_link and not os.path.exists(filepath):
        target = os.readlink(filepath)
        # Extract size from git-annex path if available
        match = re.search(r"\.git/annex.*?-s(\d+)-", target)
        if match:
            data = {"_DataLink_": f"{attach_url}{relpath2}&size={match.group(1)}"}
        else:
            data = {"_DataLink_": f"symlink:{target}"}
        _save_json(outfile + ".json", data)
        return

    # Empty file
    if filesize == 0:
        _save_json(outfile + ".json", [])
        return

    # Route to appropriate converter based on extension
    if _has_ext(filepath, ["tsv", "csv", "tsv.gz", "csv.gz"]):
        _convert_tabular(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            filesize,
            config,
            attach_url,
            fallback_url,
            outputroot,
        )
    elif _has_ext(filepath, ["json", "jmsh", "jnii", "jnirs"]):
        _convert_json_file(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            filesize,
            config,
            attach_url,
            fallback_url,
            outputroot,
        )
    elif _has_ext(filepath, ["bmsh", "bnii", "bnirs", "jdb", "bjd"]):
        _convert_binary_json(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            filesize,
            config,
            attach_url,
            outputroot,
        )
    elif _has_ext(filepath, ["bval", "bvec"]):
        _convert_bvec(
            filepath,
            outfile,
            relpath2,
            dsname,
            filesize,
            config,
            fallback_url,
            outputroot,
        )
    elif _has_ext(filepath, ["nii", "nii.gz", "hdr", "img", "img.gz"]):
        _convert_nifti(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            config,
            attach_url,
            outputroot,
        )
    elif _has_ext(filepath, ["snirf"]):
        _convert_snirf(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            config,
            attach_url,
            outputroot,
        )
    elif _has_ext(filepath, ["mat"]):
        _convert_mat(
            filepath,
            outfile,
            relpath2,
            dsname,
            contenthash,
            filesize,
            config,
            attach_url,
            outputroot,
        )
    elif _has_ext(filepath, ["txt", "md", "m", "rst"]) or _is_text_file(fname):
        _convert_text(filepath, outfile)
    elif _has_ext(filepath, ["png", "jpg", "jpeg", "gif", "pdf"]):
        _convert_attachment(
            filepath, outfile, relpath2, dsname, contenthash, attach_url, outputroot
        )
    else:
        # Default: create external link
        _save_json(
            outfile + ".json",
            {
                "_DataLink_": f"{attach_url}{dsname}&size={filesize}&file={_url_encode(relpath2)}"
            },
        )


def _url_encode(s: str) -> str:
    """URL-encode a string."""
    import urllib.parse

    return urllib.parse.quote(s, safe="")


# =============================================================================
# Type-specific converters
# =============================================================================


def _convert_tabular(
    filepath,
    outfile,
    relpath2,
    dsname,
    contenthash,
    filesize,
    config,
    attach_url,
    fallback_url,
    outputroot,
):
    """Convert TSV/CSV files (including gzipped)."""
    is_participants = "participants.tsv" in os.path.basename(filepath)

    if filesize < config["MAX_TSV"] or is_participants:
        # Inline: check for duplicate by SHA1
        sha1 = _file_hash(filepath, "sha1")
        existing = _check_duplicate(outputroot, dsname, sha1, "")
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            data = load_csv_tsv(filepath, return_dict=True, convert_numeric=True)
            _save_json(outfile + ".json", data)
            _register_hash(outputroot, dsname, sha1, "", relpath2)
    else:
        # Large file: save as attachment
        if fallback_url:
            url = fallback_url.replace(
                "%DB%", dsname.split("/")[0] if "/" in dsname else dsname
            ).replace("%DS%", dsname)
            _save_json(
                outfile + ".json",
                {"_DataLink_": f"{url}&size={filesize}&file={relpath2}"},
            )
        else:
            suffix = ".tsv.json"
            existing = _check_duplicate(outputroot, dsname, contenthash, suffix)
            if existing:
                _save_json(outfile + ".json", {"_DataLink_": existing})
            else:
                attdir = os.path.join(outputroot, ".att", dsname)
                os.makedirs(attdir, exist_ok=True)
                attfile = os.path.join(attdir, f"{contenthash}{suffix}")
                data = load_csv_tsv(filepath, return_dict=True, convert_numeric=True)
                _save_json(attfile, data)
                attsize = os.path.getsize(attfile)
                _save_json(
                    outfile + ".json",
                    {
                        "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{suffix}"
                    },
                )
                _register_hash(outputroot, dsname, contenthash, suffix, relpath2)


def _convert_json_file(
    filepath,
    outfile,
    relpath2,
    dsname,
    contenthash,
    filesize,
    config,
    attach_url,
    fallback_url,
    outputroot,
):
    """Convert JSON-based files."""
    sha1 = _file_hash(filepath, "sha1")

    if filesize < config["MAX_JSON"]:
        existing = _check_duplicate(outputroot, dsname, sha1, "")
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            data = loadt(filepath, decode=False)
            _save_json(
                outfile if outfile.endswith(".json") else outfile + ".json", data
            )
            _register_hash(outputroot, dsname, sha1, "", relpath2)
    else:
        suffix = ".json"
        existing = _check_duplicate(outputroot, dsname, contenthash, suffix)
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            attdir = os.path.join(outputroot, ".att", dsname)
            os.makedirs(attdir, exist_ok=True)
            attfile = os.path.join(attdir, f"{contenthash}{suffix}")
            data = loadt(filepath, decode=False)
            _save_json(attfile, data)
            attsize = os.path.getsize(attfile)
            _save_json(
                outfile + ".json",
                {
                    "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{suffix}"
                },
            )
            _register_hash(outputroot, dsname, contenthash, suffix, relpath2)


def _convert_binary_json(
    filepath,
    outfile,
    relpath2,
    dsname,
    contenthash,
    filesize,
    config,
    attach_url,
    outputroot,
):
    """Convert binary JSON files (bmsh, bnii, bnirs, jdb)."""
    if filesize < config["MAX_BJSON"]:
        data = load(filepath, decode=True)
        savejd(data, outfile + ".json", encode=True, compression="zlib")
    else:
        ext = os.path.splitext(filepath)[1]
        existing = _check_duplicate(outputroot, dsname, contenthash, ext)
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            attdir = os.path.join(outputroot, ".att", dsname)
            os.makedirs(attdir, exist_ok=True)
            shutil.copy2(filepath, os.path.join(attdir, f"{contenthash}{ext}"))
            _save_json(
                outfile + ".json",
                {
                    "_DataLink_": f"{attach_url}{dsname}&size={filesize}&file={contenthash}{ext}"
                },
            )
            _register_hash(outputroot, dsname, contenthash, ext, relpath2)


def _convert_bvec(
    filepath, outfile, relpath2, dsname, filesize, config, fallback_url, outputroot
):
    """Convert bval/bvec files."""
    if filesize > config["MAX_BVEC"]:
        _save_json(
            outfile + ".json",
            {"_DataLink_": f"{fallback_url}/{relpath2}&size={filesize}"},
        )
    else:
        sha1 = _file_hash(filepath, "sha1")
        existing = _check_duplicate(outputroot, dsname, sha1, "")
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            data = np.genfromtxt(filepath, dtype=np.float32)
            savejd(data, outfile + ".json", encode=True, compression="zlib")
            _register_hash(outputroot, dsname, sha1, "", relpath2)


def _convert_nifti(
    filepath, outfile, relpath2, dsname, contenthash, config, attach_url, outputroot
):
    """Convert NIfTI files (.nii, .nii.gz, .hdr)."""
    suffix = "-zlib.bnii"
    existing = _check_duplicate(outputroot, dsname, contenthash, suffix)
    if existing:
        _save_json(outfile + ".json", {"_DataLink_": existing})
        return

    attdir = os.path.join(outputroot, ".att", dsname)
    os.makedirs(attdir, exist_ok=True)
    attfile = os.path.join(attdir, f"{contenthash}{suffix}")

    nii = nii2jnii(filepath)
    if isinstance(nii, dict) and "NIFTIHeader" in nii:
        nii["NIFTIHeader"]["SrcFilePath"] = relpath2

    saveb(nii, attfile, encode=True, compression="zlib")
    attsize = os.path.getsize(attfile)

    # Replace data with links
    if "NIFTIData" in nii:
        nii["NIFTIData"] = {
            "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{suffix}:$.NIFTIData"
        }
    if "NIFTIExtension" in nii:
        nii["NIFTIExtension"] = {
            "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{suffix}:$.NIFTIExtension"
        }

    savejson(nii, outfile + ".jnii")
    _register_hash(outputroot, dsname, contenthash, suffix, relpath2)


def _convert_snirf(
    filepath, outfile, relpath2, dsname, contenthash, config, attach_url, outputroot
):
    """Convert SNIRF files."""
    suffix = "-zlib.jdb"
    existing = _check_duplicate(outputroot, dsname, contenthash, suffix)
    if existing:
        _save_json(outfile + ".json", {"_DataLink_": existing})
        return

    attdir = os.path.join(outputroot, ".att", dsname)
    os.makedirs(attdir, exist_ok=True)
    attfile = os.path.join(attdir, f"{contenthash}{suffix}")

    snirf_data = loadsnirf(filepath)
    header, binary = {}, {}

    # Extract large arrays to binary, keep small data in header
    def extract_arrays(obj, path=""):
        if not isinstance(obj, dict):
            return
        for key, val in list(obj.items()):
            newpath = f"{path}.{key}" if path else key
            is_large = (isinstance(val, (list, tuple)) and len(val) > 1000) or (
                hasattr(val, "size") and val.size > 1000
            )
            if is_large:
                binary[newpath] = val
                obj[key] = {
                    "_DataLink_": f"{attach_url}{dsname}&file={contenthash}{suffix}:$.{newpath}"
                }
            elif isinstance(val, dict):
                extract_arrays(val, newpath)

    import copy

    header = copy.deepcopy(snirf_data)
    extract_arrays(header)

    saveb(binary, attfile, encode=True, compression="zlib")
    savejson(header, outfile + ".json")
    _register_hash(outputroot, dsname, contenthash, suffix, relpath2)


def _convert_mat(
    filepath,
    outfile,
    relpath2,
    dsname,
    contenthash,
    filesize,
    config,
    attach_url,
    outputroot,
):
    """Convert MATLAB .mat files."""
    try:
        import scipy.io as sio
    except ImportError:
        print(f"scipy not available, skipping: {filepath}")
        return

    data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data = {k: v for k, v in data.items() if not k.startswith("__")}

    if filesize < config["MAX_MAT"]:
        savejd(data, outfile + ".json", encode=True, compression="zlib")
    else:
        suffix = "-zlib.jdb"
        existing = _check_duplicate(outputroot, dsname, contenthash, suffix)
        if existing:
            _save_json(outfile + ".json", {"_DataLink_": existing})
        else:
            attdir = os.path.join(outputroot, ".att", dsname)
            os.makedirs(attdir, exist_ok=True)
            attfile = os.path.join(attdir, f"{contenthash}{suffix}")
            saveb(data, attfile, encode=True, compression="zlib")
            attsize = os.path.getsize(attfile)
            link_data = {
                k: {
                    "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{suffix}:$.{k}"
                }
                for k in data
            }
            savejson(link_data, outfile + ".json")
            _register_hash(outputroot, dsname, contenthash, suffix, relpath2)


def _convert_text(filepath, outfile):
    """Convert text files (README, etc.)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    _save_json(outfile + ".json", content)


def _export_file(filepath: str, data: Any, rootdata: Dict = None) -> None:
    """
    Export a single file from JSON data, handling special formats.

    Handles:
    - TSV files with _EnumKey_/_EnumValue_ encoding
    - _DataLink_ references (internal JSONPath or external URLs)
    - Regular JSON/text data

    Args:
        filepath: Output file path
        data: Data to export
        rootdata: Root dataset for resolving internal JSONPath references
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    filepath_lower = filepath.lower()

    # Handle _DataLink_
    if isinstance(data, dict) and "_DataLink_" in data and len(data) == 1:
        link = data["_DataLink_"]
        _resolve_datalink(link, filepath, rootdata)
        return

    # Handle TSV files - decode enum encoding
    if filepath_lower.endswith(".tsv"):
        if isinstance(data, dict):
            _save_as_tsv(filepath, data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data) if data else "")
        return

    # Handle CSV files
    if filepath_lower.endswith(".csv"):
        if isinstance(data, dict):
            _save_as_csv(filepath, data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data) if data else "")
        return

    # Handle JSON files
    if filepath_lower.endswith((".json", ".jnii", ".jnirs")):
        _save_json(filepath, data)
        return

    # Handle text content (strings)
    if isinstance(data, str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
        return

    # Handle binary content
    if isinstance(data, (bytes, bytearray)):
        with open(filepath, "wb") as f:
            f.write(data)
        return

    # Default: save as JSON
    _save_json(filepath, data)


def _resolve_datalink(link: str, filepath: str, rootdata: Dict = None) -> None:
    """
    Resolve a _DataLink_ reference.

    Handles:
    - Internal JSONPath references ($.path.to.data)
    - Symlinks (symlink:target)
    - External URLs (https://...)
    """
    if link.startswith("$."):
        # Internal JSONPath reference - create symlink or resolve data
        if rootdata:
            jsonpath_str = link
            # Convert JSONPath to relative file path for symlink
            path_part = jsonpath_str[2:]  # Remove $.
            placeholder = "_0x2E_"
            path_part = path_part.replace("\\.", placeholder)
            parts = path_part.split(".")
            parts = [p.replace(placeholder, ".") for p in parts]

            if parts:
                # Try to create relative symlink
                relpath = os.path.join(*parts)
                dest_dir = os.path.dirname(filepath)
                export_root = dest_dir
                # Walk up to find export root (heuristic: look for dataset_description.json)
                while export_root and not os.path.exists(
                    os.path.join(export_root, "dataset_description.json")
                ):
                    parent = os.path.dirname(export_root)
                    if parent == export_root:
                        break
                    export_root = parent

                target_path = (
                    os.path.join(export_root, relpath) if export_root else relpath
                )
                try:
                    rel_target = os.path.relpath(target_path, dest_dir)
                    if os.path.exists(filepath) or os.path.islink(filepath):
                        os.remove(filepath)
                    os.symlink(rel_target, filepath)
                    return
                except (ValueError, OSError) as e:
                    print(f"Warning: Could not create symlink {filepath}: {e}")

    elif link.startswith("symlink:"):
        # Explicit symlink
        target = link[8:]  # Remove "symlink:"
        try:
            if os.path.exists(filepath) or os.path.islink(filepath):
                os.remove(filepath)
            os.symlink(target, filepath)
        except OSError as e:
            print(f"Warning: Could not create symlink {filepath}: {e}")
        return

    elif link.startswith(("http://", "https://")):
        # External URL - download
        _download_file(link, filepath)
        return

    print(f"Warning: Unknown _DataLink_ format: {link}")


def _download_file(url: str, filepath: str) -> None:
    """Download file from URL, following redirects."""
    import urllib.request
    import urllib.error

    # Extract JSONPath fragment if present (url:$.path)
    if ":$." in url:
        url = url.split(":$.")[0]

    print(f"Downloading: {url}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        # Create request with redirect handling
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "neuroj/1.0")

        with urllib.request.urlopen(req) as response:
            content = response.read()

        # Determine if binary based on extension
        is_binary = filepath.lower().endswith(
            (
                ".bnii",
                ".bmsh",
                ".bnirs",
                ".jdb",
                ".bjd",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".pdf",
                ".nii",
                ".gz",
                ".snirf",
                ".mat",
            )
        )

        mode = "wb" if is_binary else "w"
        with open(filepath, mode) as f:
            if is_binary:
                f.write(content)
            else:
                f.write(content.decode("utf-8", errors="replace"))

        print(f"  Saved: {filepath}")

    except urllib.error.HTTPError as e:
        print(f"Warning: HTTP error downloading {url}: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"Warning: URL error downloading {url}: {e.reason}")
    except Exception as e:
        print(f"Warning: Error downloading {url}: {e}")


def _convert_attachment(
    filepath, outfile, relpath2, dsname, contenthash, attach_url, outputroot
):
    """Copy binary files (images, PDFs) as attachments."""
    ext = _get_extension(filepath) or os.path.splitext(filepath)[1]
    existing = _check_duplicate(outputroot, dsname, contenthash, ext)
    if existing:
        _save_json(outfile + ".json", {"_DataLink_": existing})
    else:
        attdir = os.path.join(outputroot, ".att", dsname)
        os.makedirs(attdir, exist_ok=True)
        attfile = os.path.join(attdir, f"{contenthash}{ext}")
        shutil.copy2(filepath, attfile)
        attsize = os.path.getsize(attfile)
        _save_json(
            outfile + ".json",
            {
                "_DataLink_": f"{attach_url}{dsname}&size={attsize}&file={contenthash}{ext}"
            },
        )
        _register_hash(outputroot, dsname, contenthash, ext, relpath2)


# =============================================================================
# JSON merging (post-processing)
# =============================================================================


def _mergejson(folder: str) -> None:
    """
    Merge all JSON files in subfolders into a single .json file per subfolder.
    Mimics the mergejson shell script.
    """
    print(f"Merging JSON files in {folder}")

    for subfolder in glob.glob(os.path.join(folder, "*/")):
        subfolder_name = os.path.basename(subfolder.rstrip("/\\"))
        if subfolder_name.startswith("."):
            continue

        result = {}
        for root, dirs, files in os.walk(subfolder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if not fname.endswith((".json", ".jnii", ".jnirs")):
                    continue
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, subfolder)

                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Remove extension and build nested structure
                    key = re.sub(r"\.(jnii|jnirs|json)$", "", relpath)
                    parts = key.replace("\\", "/").split("/")
                    current = result
                    for part in parts[:-1]:
                        current = current.setdefault(part, {})
                    current[parts[-1]] = data
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")

        if result:
            with open(os.path.join(folder, f"{subfolder_name}.json"), "w") as f:
                json.dump(result, f, separators=(",", ":"))


def _bids2json(outputroot: str, dsname: str) -> None:
    """
    Merge all JSON files into a single dataset JSON file.
    Mimics the bids2json shell script.
    """
    print(f"Creating {dsname}.json")
    dspath = os.path.join(outputroot, dsname)
    result = {}

    for item in os.listdir(dspath):
        if item.startswith("."):
            continue
        itempath = os.path.join(dspath, item)
        if os.path.isfile(itempath) and item.endswith((".json", ".jnii", ".jnirs")):
            try:
                with open(itempath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = re.sub(r"\.(jnii|jnirs|json)$", "", item)
                result[key] = data
            except Exception as e:
                print(f"Error reading {itempath}: {e}")

    with open(os.path.join(outputroot, f"{dsname}.json"), "w") as f:
        json.dump(result, f, separators=(",", ":"))
