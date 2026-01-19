"""@package docstring
REST-API interface to NeuroJSON.io

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "neuroj",
    "neurojgui",
]

##====================================================================================
## dependent libraries
##====================================================================================
import os
import re
import urllib.request
import urllib.parse
import urllib.error
import json
import csv
import shutil
from typing import Dict, Any, Tuple, Union
import time
from .jfile import jdlink
from .jpath import jsonpath


def neuroj(
    cmd: str,
    db: str = None,
    ds: str = None,
    file: str = None,
    *varargin,
    **kwargs,
) -> Union[Tuple[Any, str, str], Any]:
    """
    NeuroJSON.io client - browsing/listing/downloading/uploading data
    provided at https://neurojson.io

    author: Qianqian Fang (q.fang <at> neu.edu)

    Args:
        cmd: a string, must be one of
               'gui':
                  - start a GUI and interactively browse datasets
               'list':
                  - if followed by nothing, list all databases
                  - if database is given, list its all datasets
                  - if dataset is given, list all its revisions
               'info': return metadata associated with the specified
                     database, dataset or file of a dataset
               'get': must provide database and dataset name, download and
                     parse the specified dataset or its file
               'export': export a dataset to a local folder structure
               'find':
                  - if database is a string '/.../', find database by a
                    regular expression pattern
                  - if database is a dict, find database using
                    NeuroJSON's search API
                  - if dataset is a string '/.../', find datasets by a
                    regular expression pattern
                  - if dataset is a dict, find database using
                    the _find API

            admin commands (require database admin credentials):
               'put': create database, create dataset under a dataset, or
                     upload an file under a dataset
               'delete': delete the specified file, dataset or
                     database
        db: database name
        ds: dataset name
        file: attachment file name
        **kwargs: additional keyward parameters, including
            limit: limit the returned response items
            skip: offset of the retrieved record in the database
            raw: if set to True, return (res, restapi, jsonstring), otherwise, only return res
            exportpath: for 'export' command, specify the export folder

    Returns:
        res: parsed response data
        restapi: the URL or REST-API of the desired resource (only when raw=True is used)
        jsonstring: the JSON raw data from the URL (only when raw=True is used)

    Example:
        neuroj('gui')  # start neuroj client in the GUI mode

        res = neuroj('list')  # list all databases under res['database']
        res = neuroj('list', 'cotilab')  # list all dataset under res['dataset']
        res = neuroj('list', 'cotilab', 'CSF_Neurophotonics_2025')  # list all versions
        res = neuroj('info')  # list metadata of all datasets
        res = neuroj('info', 'cotilab')  # list metadata of a given database
        res = neuroj('info', 'cotilab', 'CSF_Neurophotonics_2025')  # list dataset header
        res, url, rawstr = neuroj('get', db='mcx', ds='colin27', raw=True)
        res = neuroj('get', 'cotilab', 'CSF_Neurophotonics_2025')
        res = neuroj('export', 'bfnirs', 'Motor-Orihuela-Espina2010', exportpath='/tmp')

    License:
        BSD or GPL version 3, see LICENSE_{BSD,GPLv3}.txt files for details

    This function is part of JSONLab toolbox (http://iso2mesh.sf.net/cgi-bin/index.cgi?jsonlab)
    """

    if not cmd:
        print("NeuroJSON.io Client (https://neurojson.io)")
        print(
            "Format:\n\tdata = neuroj(command, ds=database, ds=datasetname, file=attachment, ...)\n"
        )
        return None

    if len(varargin) == 0 and cmd == "gui":
        # Start GUI
        gui = neurojgui()
        gui.run()
        return None

    opt = {}
    if len(varargin):
        # Convert remaining args to options dict
        args = list(varargin)
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                opt[args[i]] = args[i + 1]

    kwargs.update(opt)

    # Get server URL
    serverurl = os.getenv("NEUROJSON_IO")
    if not serverurl:
        serverurl = "https://neurojson.io:7777/"

    serverurl = kwargs.get("server", serverurl)
    options = kwargs.get("weboptions", {})
    kwargs["weboptions"] = options
    rev = kwargs.get("rev", "")

    cmd = cmd.lower()
    restapi = serverurl

    if cmd == "list":
        restapi = f"{serverurl}sys/registry"
        if db:
            restapi = f"{serverurl}{db}/_all_docs"
            if ds:
                restapi = f"{serverurl}{db}/{ds}?revs_info=true"

        jsonstring = _load_json_raw(restapi, **kwargs)
        res = json.loads(jsonstring)

        if ds:
            res = res["_revs_info"]
        elif db:
            res["dataset"] = res["rows"]
            del res["rows"]

    elif cmd == "info":
        restapi = f"{serverurl}_dbs_info"
        if db:
            restapi = f"{serverurl}{db}/"
            if ds:
                restapi = f"{serverurl}{db}/{ds}"
                if file:
                    restapi = f"{serverurl}{db}/{ds}/{file}"

        if ds or file:
            res = _load_json_header(restapi, **kwargs)
            jsonstring = json.dumps(res)
        else:
            jsonstring = _load_json_raw(restapi, **kwargs)
            res = json.loads(jsonstring)

    elif cmd == "get":
        if not ds:
            raise ValueError("get requires a dataset, i.e. document, name")

        if not file:
            restapi = f"{serverurl}{db}/{ds}"
            if rev:
                restapi = f"{serverurl}{db}/{ds}?rev={rev}"
        else:
            restapi = f"{serverurl}{db}/{ds}/{file}"

        # jdlink returns (data, cachefile) - but data may be None due to downloadonly logic
        result = jdlink(restapi)

        # Handle different return formats from jdlink
        if isinstance(result, tuple):
            res, cachefile = result[0], result[1]
        else:
            res = result
            cachefile = None

        # If res is None but cachefile exists, load from cachefile
        if res is None and cachefile:
            if isinstance(cachefile, list):
                cachefile = cachefile[0]
            if cachefile and os.path.exists(cachefile):
                with open(cachefile, "r", encoding="utf-8") as f:
                    try:
                        res = json.load(f)
                    except json.JSONDecodeError:
                        # Try binary format or return raw content
                        f.seek(0)
                        res = f.read()

        jsonstring = (
            json.dumps(res)
            if isinstance(res, (dict, list))
            else str(res)
            if res
            else ""
        )

    elif cmd == "export":
        if not ds:
            raise ValueError("export requires a dataset name")

        # Get export path
        exportroot = kwargs.get("exportpath", None)
        if not exportroot:
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                exportroot = filedialog.askdirectory(
                    title="Select folder to export dataset"
                )
                root.destroy()
                if not exportroot:
                    return None
            except:
                raise ValueError("exportpath must be specified")

        # Load the dataset
        restapi = f"{serverurl}{db}/{ds}"
        if rev:
            restapi = f"{serverurl}{db}/{ds}?rev={rev}"

        # Get data using the same logic as 'get' command
        result = jdlink(restapi)

        if isinstance(result, tuple):
            data, jsonfile = result[0], result[1]
        else:
            data = result
            jsonfile = None

        # If data is None but jsonfile exists, load from jsonfile
        if data is None and jsonfile:
            if isinstance(jsonfile, list):
                jsonfile = jsonfile[0]
            if jsonfile and os.path.exists(jsonfile):
                with open(jsonfile, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        pass

        if isinstance(jsonfile, list):
            jsonfile = jsonfile[0]

        # Check if BIDS dataset - look for dataset_description.json with BIDSVersion
        is_bids = False
        if isinstance(data, dict):
            # Try different possible key formats
            dd_keys = ["dataset_description.json", "dataset_description_json"]
            for dd_key in dd_keys:
                if dd_key in data:
                    dd_content = data[dd_key]
                    if isinstance(dd_content, dict) and "BIDSVersion" in dd_content:
                        is_bids = True
                        print(
                            f"DEBUG: Found BIDS dataset (key={dd_key}, BIDSVersion={dd_content['BIDSVersion']})"
                        )
                        break

            if not is_bids:
                print(
                    f"DEBUG: Not a BIDS dataset. Keys in data: {list(data.keys())[:10]}"
                )
                for k in data.keys():
                    if "dataset_description" in k.lower():
                        print(
                            f"DEBUG: Found key containing 'dataset_description': {k} = {data[k]}"
                        )

        if is_bids:
            # Create dataset subfolder and perform folder reconstruction
            datasetfolder = os.path.join(exportroot, ds)
            os.makedirs(datasetfolder, exist_ok=True)
            _export_data(data, datasetfolder, ds, data, jsonfile)
            res = {"exportpath": datasetfolder, "status": "success", "type": "BIDS"}
        else:
            # Not a BIDS dataset - just copy the cached JSON file
            if jsonfile:
                _, fext = os.path.splitext(jsonfile)
                if not fext:
                    fext = ".json"
                destfile = os.path.join(exportroot, ds + fext)
                shutil.copy2(jsonfile, destfile)
                res = {"exportpath": destfile, "status": "success", "type": "JSON"}
            else:
                # No cache file, save data directly
                destfile = os.path.join(exportroot, ds + ".json")
                with open(destfile, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                res = {"exportpath": destfile, "status": "success", "type": "JSON"}

        jsonstring = json.dumps(res)

    elif cmd == "put":
        if not db:
            raise ValueError("put requires at least a database name")

        restapi = f"{serverurl}{db}"

        if ds:
            if not file:
                raise ValueError("must provide JSON input to upload")

            if isinstance(file, str) and os.path.exists(file):
                # File upload
                afpath, afname = os.path.split(file)
                attname = kwargs.get("filename", afname)
                restapi = f"{serverurl}{db}/{ds}/{attname}"
                res = _web_upload(restapi, file, options)
            else:
                # JSON data upload
                restapi = f"{serverurl}{db}/_design/qq/_update/timestamp/{ds}"
                jsonstring = json.dumps(file, separators=(",", ":"))
                res = _web_write(
                    restapi,
                    jsonstring,
                    {"method": "POST", "content_type": "application/json", **options},
                )
        else:
            res = _web_write(restapi, "", {"method": "PUT", **options})

    elif cmd == "delete":
        if not db:
            raise ValueError("delete requires at least a database name")

        restapi = f"{serverurl}{db}"
        if ds:
            restapi = f"{serverurl}{db}/{ds}"
            if file:
                restapi = f"{serverurl}{db}/{ds}/{file}"

        res = _web_write(restapi, "", {"method": "DELETE", **options})

    elif cmd == "find":
        if not db:
            raise ValueError(
                "find requires at least a search regular expression pattern"
            )

        if isinstance(db, str) and db.startswith("/") and db.endswith("/"):
            # Find databases by regex
            dblist, restapi, jsonstring = neuroj("list", raw=True)
            res = []
            pattern = db[1:-1]  # Remove leading and trailing /
            for dbitem in dblist["database"]:
                if re.search(pattern, json.dumps(dbitem), re.IGNORECASE):
                    res.append(dbitem)

        elif isinstance(db, dict):
            # Find using search API
            params = "&".join([f"{k}={v}" for k, v in db.items()])
            restapi = f"https://neurojson.org/io/search.cgi?{params}"
            jsonstring = _load_json_raw(restapi, **kwargs)
            res = json.loads(jsonstring)

        elif ds and isinstance(ds, str) and ds.startswith("/") and ds.endswith("/"):
            # Find datasets by regex
            dslist, restapi, jsonstring = neuroj("list", db, raw=True)
            res = []
            pattern = ds[1:-1]
            for dataset in dslist["dataset"]:
                if re.search(pattern, dataset["id"], re.IGNORECASE):
                    res.append(dataset["id"])

        elif ds and (
            isinstance(ds, dict)
            or (isinstance(ds, str) and ds.startswith("{") and ds.endswith("}"))
        ):
            # Find using _find API
            restapi = f"{serverurl}{db}/_find"

            if isinstance(ds, dict):
                if "selector" not in ds:
                    ds["selector"] = {}
                payload = json.dumps(ds, separators=(",", ":"))
            else:
                payload = ds

            res = _web_write(
                restapi,
                payload,
                {"method": "POST", "content_type": "application/json", **options},
            )

    else:
        raise ValueError(f"Unknown command: {cmd}")

    # Return based on context
    if "jsonstring" in locals() and kwargs.get("raw", False):
        return res, restapi, jsonstring
    else:
        return res


##====================================================================================
## Export helper functions - replace existing ones with these
##====================================================================================


def _export_data(data, currentfolder, parentkey, rootdata, cachefile, exportroot=None):
    """Export data structure to folder hierarchy"""

    if exportroot is None:
        exportroot = currentfolder

    if not isinstance(data, dict):
        return

    datainfo = {}

    for key, val in data.items():
        decodedkey = key

        # Check if it's a file (contains . with suffix, not starting with .)
        if re.search(r"\.[^\.\/\\]+$", decodedkey) and not decodedkey.startswith("."):
            filepath = os.path.join(currentfolder, decodedkey)

            # .snirf file with SNIRFData
            if (
                decodedkey.lower().endswith(".snirf")
                and isinstance(val, dict)
                and "SNIRFData" in val
            ):
                try:
                    from .jfile import savesnirf

                    savesnirf(val["SNIRFData"], filepath)
                except:
                    with open(filepath, "w") as f:
                        json.dump(val, f, indent=2)

            # .tsv file - convert JSON to TSV
            elif decodedkey.lower().endswith(".tsv") and isinstance(val, dict):
                _save_struct_to_tsv(val, filepath)

            # _DataLink_ (internal or external)
            elif isinstance(val, dict) and "_DataLink_" in val:
                linkurl = val["_DataLink_"]
                if linkurl:
                    if linkurl.startswith("$"):
                        _resolve_internal(rootdata, linkurl, filepath, exportroot)
                    else:
                        _, cachedfile = jdlink(linkurl)
                        if isinstance(cachedfile, list):
                            cachedfile = cachedfile[0]
                        if cachedfile:
                            _create_link(cachedfile, filepath)

            elif isinstance(val, str):
                with open(filepath, "w") as f:
                    f.write(val)

            elif isinstance(val, (bytes, bytearray)):
                with open(filepath, "wb") as f:
                    f.write(val)

            elif isinstance(val, dict):
                with open(filepath, "w") as f:
                    json.dump(val, f, indent=2)

            else:
                with open(filepath, "w") as f:
                    json.dump(val, f, indent=2)

        # _DataLink_ without file extension
        elif isinstance(val, dict) and len(val) == 1 and "_DataLink_" in val:
            linkurl = val["_DataLink_"]
            if linkurl:
                linkpath = os.path.join(currentfolder, decodedkey)
                if linkurl.startswith("$"):
                    _resolve_internal(rootdata, linkurl, linkpath, exportroot)
                else:
                    _, cachedfile = jdlink(linkurl)
                    if isinstance(cachedfile, list):
                        cachedfile = cachedfile[0]
                    if cachedfile:
                        _create_link(cachedfile, linkpath)

        # Metadata fields for .datainfo.json
        elif (
            decodedkey in ("_id", "_rev")
            or decodedkey.startswith("Mesh")
            or re.match(r"^_Array.*_$", decodedkey)
        ):
            datainfo[key] = val

        # Subfolder
        elif isinstance(val, dict):
            subfolder = os.path.join(currentfolder, decodedkey)
            os.makedirs(subfolder, exist_ok=True)
            _export_data(val, subfolder, decodedkey, rootdata, cachefile, exportroot)

    if datainfo:
        with open(os.path.join(currentfolder, ".datainfo.json"), "w") as f:
            json.dump(datainfo, f, indent=2)


def _resolve_internal(rootdata, jsonpath_str, destpath, exportroot=None):
    """Resolve internal JSONPath reference - create relative symlink or save the result"""

    try:
        # Convert JSONPath to relative file path for symlink
        # Example: $.sub-6022.ses-1.nirs.sub-6022_ses-1_task-MA_run-01_channels\.tsv
        # becomes: sub-6022/ses-1/nirs/sub-6022_ses-1_task-MA_run-01_channels.tsv

        if jsonpath_str.startswith("$.") and exportroot:
            # Remove $. prefix
            path_part = jsonpath_str[2:]

            # Use same logic as jsonpath.py: replace \. with placeholder
            placeholder = "_0x2E_"
            path_part = path_part.replace("\\.", placeholder)

            # Split by dots (path separators)
            parts = path_part.split(".")

            # Restore dots in each part (convert placeholder back to .)
            parts = [p.replace(placeholder, ".") for p in parts]

            # Build relative path
            if parts:
                relpath = os.path.join(*parts)
                target_path = os.path.join(exportroot, relpath)

                # Calculate relative path from destpath's directory to target
                dest_dir = os.path.dirname(destpath)
                try:
                    rel_target = os.path.relpath(target_path, dest_dir)

                    # Create parent directories if needed
                    os.makedirs(dest_dir, exist_ok=True)

                    # Create symlink with relative path
                    if os.path.exists(destpath) or os.path.islink(destpath):
                        os.remove(destpath)
                    os.symlink(rel_target, destpath)
                    return
                except (ValueError, OSError) as e:
                    print(
                        f"Warning: Could not create symlink {destpath} -> {rel_target}: {e}"
                    )
                    pass  # Fall through to resolve data

        # Fallback: resolve and save actual data using jsonpath
        from .jpath import jsonpath

        resolved = jsonpath(rootdata, jsonpath_str)

        if resolved is None:
            print(f"Warning: Could not resolve jsonpath: {jsonpath_str}")
            return

        _, ext = os.path.splitext(destpath)

        if ext.lower() == ".tsv" and isinstance(resolved, dict):
            _save_struct_to_tsv(resolved, destpath)
        elif isinstance(resolved, str):
            with open(destpath, "w") as f:
                f.write(resolved)
        elif isinstance(resolved, dict):
            with open(destpath, "w") as f:
                json.dump(resolved, f, indent=2)
        elif isinstance(resolved, (bytes, bytearray)):
            with open(destpath, "wb") as f:
                f.write(resolved)
        else:
            with open(destpath, "w") as f:
                json.dump(resolved, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not resolve internal link: {jsonpath_str} - {e}")


def _save_struct_to_tsv(data, filepath):
    """Convert a dict with column arrays to TSV format"""

    if not isinstance(data, dict) or not data:
        return

    keys = list(data.keys())
    firstcol = data[keys[0]]

    if isinstance(firstcol, list):
        nrows = len(firstcol)
    else:
        nrows = 1

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Write header
        writer.writerow(keys)

        # Write data rows
        for r in range(nrows):
            row = []
            for key in keys:
                coldata = data[key]
                if isinstance(coldata, list) and r < len(coldata):
                    val = coldata[r]
                else:
                    val = coldata
                row.append(str(val) if val is not None else "")
            writer.writerow(row)


def _create_link(target, linkname):
    """Create a symbolic link (platform-dependent)"""

    try:
        if os.path.exists(linkname):
            os.remove(linkname)
        os.symlink(target, linkname)
    except OSError:
        # Fall back to copying if linking fails
        shutil.copy2(target, linkname)


##====================================================================================
## GUI class
##====================================================================================


class neurojgui:
    """
    GUI class for NeuroJSON browser with search panel and export
    """

    # Icon definitions (base64 encoded PNG or simple text icons)
    ICONS = {
        "database": "📁",
        "folder": "📂",
        "jdata": "🔢",
        "mesh": "🔷",
        "nifti": "🧠",
        "snirf": "💡",
        "data": "📄",
        "parent": "⬆️",
        "link": "🔗",
        "refresh": "🔄",
        "search": "🔍",
        "export": "📤",
        "clear": "🗑️",
        "close": "❌",
    }

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.filedialog = filedialog

        self.root = self.tk.Tk()
        self.root.title("NeuroJSON.io Dataset Browser")
        self.root.geometry("1000x700")

        # Initialize data storage
        self.current_db = ""
        self.current_ds = ""
        self.datasets = None
        self.rootpath = ""
        self.search_datasets = {}
        self.search_subjects = {}
        self.search_visible = False

        # Create frames
        self.create_widgets()
        self.t0 = time.time()

    def create_widgets(self):
        """Create GUI widgets"""

        # Create toolbar frame
        toolbar = self.tk.Frame(self.root, bd=1, relief=self.tk.RAISED)
        toolbar.pack(side=self.tk.TOP, fill=self.tk.X)

        # Toolbar buttons with icons
        btn_load_db = self.tk.Button(
            toolbar, text=f"{self.ICONS['refresh']} List Databases", command=self.loaddb
        )
        btn_load_db.pack(side=self.tk.LEFT, padx=5, pady=5)

        btn_search = self.tk.Button(
            toolbar, text=f"{self.ICONS['search']} Search", command=self.toggle_search
        )
        btn_search.pack(side=self.tk.LEFT, padx=5, pady=5)

        btn_export = self.tk.Button(
            toolbar, text=f"{self.ICONS['export']} Export", command=self.export_dataset
        )
        btn_export.pack(side=self.tk.LEFT, padx=5, pady=5)

        # Create search panel (initially hidden)
        self.search_frame = self.tk.LabelFrame(
            self.root, text="Dataset Search", padx=10, pady=10
        )

        # Search panel contents - Row 1
        row1 = self.tk.Frame(self.search_frame)
        row1.pack(fill=self.tk.X, pady=2)

        self.tk.Label(row1, text="Keyword:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_keyword = self.tk.Entry(row1, width=20)
        self.search_keyword.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row1, text="Database:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_database = self.ttk.Combobox(
            row1,
            values=[
                "any",
                "openneuro",
                "abide",
                "abide2",
                "datalad-registry",
                "adhd200",
            ],
            width=15,
        )
        self.search_database.set("any")
        self.search_database.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row1, text="Dataset:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_dataset = self.tk.Entry(row1, width=15)
        self.search_dataset.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row1, text="Subject:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_subject = self.tk.Entry(row1, width=15)
        self.search_subject.pack(side=self.tk.LEFT, padx=5)

        # Search panel contents - Row 2
        row2 = self.tk.Frame(self.search_frame)
        row2.pack(fill=self.tk.X, pady=2)

        self.tk.Label(row2, text="Gender:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_gender = self.ttk.Combobox(
            row2, values=["any", "male", "female", "unknown"], width=10
        )
        self.search_gender.set("any")
        self.search_gender.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row2, text="Modality:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_modality = self.ttk.Combobox(
            row2,
            values=[
                "any",
                "anat",
                "func",
                "dwi",
                "fmap",
                "perf",
                "meg",
                "eeg",
                "ieeg",
                "beh",
                "pet",
                "micr",
                "nirs",
                "motion",
            ],
            width=10,
        )
        self.search_modality.set("any")
        self.search_modality.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row2, text="Age min:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_age_min = self.tk.Entry(row2, width=8)
        self.search_age_min.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row2, text="Age max:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_age_max = self.tk.Entry(row2, width=8)
        self.search_age_max.pack(side=self.tk.LEFT, padx=5)

        # Search panel contents - Row 3
        row3 = self.tk.Frame(self.search_frame)
        row3.pack(fill=self.tk.X, pady=2)

        self.tk.Label(row3, text="Task name:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_task_name = self.tk.Entry(row3, width=15)
        self.search_task_name.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row3, text="Session:", width=10, anchor="e").pack(
            side=self.tk.LEFT
        )
        self.search_session_name = self.tk.Entry(row3, width=15)
        self.search_session_name.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row3, text="Limit:", width=10, anchor="e").pack(side=self.tk.LEFT)
        self.search_limit = self.tk.Entry(row3, width=8)
        self.search_limit.insert(0, "25")
        self.search_limit.pack(side=self.tk.LEFT, padx=5)

        self.tk.Label(row3, text="Skip:", width=10, anchor="e").pack(side=self.tk.LEFT)
        self.search_skip = self.tk.Entry(row3, width=8)
        self.search_skip.insert(0, "0")
        self.search_skip.pack(side=self.tk.LEFT, padx=5)

        # Search panel buttons
        btn_frame = self.tk.Frame(self.search_frame)
        btn_frame.pack(fill=self.tk.X, pady=10)

        self.tk.Button(
            btn_frame, text=f"{self.ICONS['search']} Search", command=self.do_search
        ).pack(side=self.tk.LEFT, padx=5)
        self.tk.Button(
            btn_frame, text=f"{self.ICONS['clear']} Clear", command=self.clear_search
        ).pack(side=self.tk.LEFT, padx=5)
        self.tk.Button(
            btn_frame, text=f"{self.ICONS['close']} Close", command=self.toggle_search
        ).pack(side=self.tk.LEFT, padx=5)

        # Create main frame
        self.main_frame = self.tk.Frame(self.root)
        self.main_frame.pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)

        # Database listbox with scrollbar
        db_frame = self.tk.Frame(self.main_frame)
        db_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH)

        self.tk.Label(db_frame, text="Database").pack()
        db_scroll = self.tk.Scrollbar(db_frame)
        db_scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        self.lsDb = self.tk.Listbox(db_frame, width=20, yscrollcommand=db_scroll.set)
        self.lsDb.pack(fill=self.tk.BOTH, expand=True)
        db_scroll.config(command=self.lsDb.yview)
        self.lsDb.bind("<Double-Button-1>", self.loadds)
        self.lsDb.bind("<Return>", self.loadds)

        # Dataset listbox with scrollbar
        ds_frame = self.tk.Frame(self.main_frame)
        ds_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH)

        self.tk.Label(ds_frame, text="Dataset").pack()
        ds_scroll = self.tk.Scrollbar(ds_frame)
        ds_scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        self.lsDs = self.tk.Listbox(ds_frame, width=30, yscrollcommand=ds_scroll.set)
        self.lsDs.pack(fill=self.tk.BOTH, expand=True)
        ds_scroll.config(command=self.lsDs.yview)
        self.lsDs.bind("<Double-Button-1>", self.loaddsdata)
        self.lsDs.bind("<Return>", self.loaddsdata)

        # JSON tree listbox with scrollbar
        json_frame = self.tk.Frame(self.main_frame)
        json_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        self.tk.Label(json_frame, text="Data").pack()
        json_scroll = self.tk.Scrollbar(json_frame)
        json_scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        self.lsJSON = self.tk.Listbox(json_frame, yscrollcommand=json_scroll.set)
        self.lsJSON.pack(fill=self.tk.BOTH, expand=True)
        json_scroll.config(command=self.lsJSON.yview)
        self.lsJSON.bind("<Double-Button-1>", self.expandjsontree)
        self.lsJSON.bind("<Return>", self.expandjsontree)

        # Value text area with scrollbar
        value_frame = self.tk.Frame(self.main_frame)
        value_frame.pack(side=self.tk.RIGHT, fill=self.tk.BOTH, expand=True)

        self.tk.Label(value_frame, text="Value").pack()
        value_scroll = self.tk.Scrollbar(value_frame)
        value_scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        self.txValue = self.tk.Text(
            value_frame, width=40, yscrollcommand=value_scroll.set
        )
        self.txValue.pack(fill=self.tk.BOTH, expand=True)
        value_scroll.config(command=self.txValue.yview)

    def run(self):
        """Start the GUI"""
        self.root.mainloop()

    def get_icon_text(self, icontype, text):
        """Get display text with icon"""
        icon = self.ICONS.get(icontype, self.ICONS["data"])
        return f"{icon} {text}"

    def detect_data_type(self, dataobj, key):
        """Detect the type of data for icon display"""
        if key == "..":
            return "parent"

        if not isinstance(dataobj, dict):
            return "data"

        fields = list(dataobj.keys())

        if len(fields) == 1 and "_DataLink_" in fields:
            return "link"

        if "_ArrayType_" in fields or "_ArraySize_" in fields:
            return "jdata"

        if "_MeshNode_" in fields or "MeshNode3D" in fields:
            return "mesh"

        if "NIFTIData" in fields or "NIFTIHeader" in fields:
            return "nifti"

        if "SNIRFData" in fields or "nirs" in fields:
            return "snirf"

        return "folder"

    def toggle_search(self):
        """Toggle search panel visibility"""
        if self.search_visible:
            self.search_frame.pack_forget()
            self.search_visible = False
        else:
            self.search_frame.pack(
                side=self.tk.TOP, fill=self.tk.X, before=self.main_frame
            )
            self.search_visible = True

    def clear_search(self):
        """Clear all search fields"""
        self.search_keyword.delete(0, self.tk.END)
        self.search_database.set("any")
        self.search_dataset.delete(0, self.tk.END)
        self.search_subject.delete(0, self.tk.END)
        self.search_gender.set("any")
        self.search_modality.set("any")
        self.search_age_min.delete(0, self.tk.END)
        self.search_age_max.delete(0, self.tk.END)
        self.search_task_name.delete(0, self.tk.END)
        self.search_session_name.delete(0, self.tk.END)
        self.search_limit.delete(0, self.tk.END)
        self.search_limit.insert(0, "25")
        self.search_skip.delete(0, self.tk.END)
        self.search_skip.insert(0, "0")

    def do_search(self):
        """Execute search query"""
        queryurl = "https://neurojson.org/io/search.cgi?"

        keyword = self.search_keyword.get().strip()
        if keyword:
            queryurl += f"keyword={urllib.parse.quote(keyword)}&"

        database = self.search_database.get()
        if database != "any":
            queryurl += f"dbname={database}&"

        dataset = self.search_dataset.get().strip()
        if dataset:
            queryurl += f"dsname={urllib.parse.quote(dataset)}&"

        subject = self.search_subject.get().strip()
        if subject:
            queryurl += f"subname={urllib.parse.quote(subject)}&"

        gender = self.search_gender.get()
        if gender != "any":
            queryurl += f"gender={gender[0]}&"

        modality = self.search_modality.get()
        if modality != "any":
            queryurl += f"modality={modality}&"

        age_min = self.search_age_min.get().strip()
        if age_min:
            try:
                queryurl += f"agemin={int(float(age_min) * 100):05d}&"
            except:
                pass

        age_max = self.search_age_max.get().strip()
        if age_max:
            try:
                queryurl += f"agemax={int(float(age_max) * 100):05d}&"
            except:
                pass

        task_name = self.search_task_name.get().strip()
        if task_name:
            queryurl += f"task={urllib.parse.quote(task_name)}&"

        session_name = self.search_session_name.get().strip()
        if session_name:
            queryurl += f"session={urllib.parse.quote(session_name)}&"

        limit = self.search_limit.get().strip() or "25"
        skip = self.search_skip.get().strip() or "0"
        queryurl += f"limit={limit}&skip={skip}"

        try:
            response = _load_json_raw(queryurl)
            result = json.loads(response)

            if not result:
                self.txValue.delete("1.0", self.tk.END)
                self.txValue.insert("1.0", "No results found")
                return

            # Process results
            unique_db = []
            datasets_by_db = {}
            subject_map = {}

            for item in result:
                if "dbname" in item and "dsname" in item:
                    dbname = item["dbname"]
                    dsname = item["dsname"]
                    key = f"{dbname}/{dsname}"

                    if dbname not in unique_db:
                        unique_db.append(dbname)
                        datasets_by_db[dbname] = []

                    if dsname not in datasets_by_db[dbname]:
                        datasets_by_db[dbname].append(dsname)

                    if key not in subject_map:
                        subject_map[key] = []

                    if "subj" in item:
                        subjname = item["subj"]
                        if subjname not in subject_map[key]:
                            subject_map[key].append(subjname)

            # Update listboxes
            self.lsDb.delete(0, self.tk.END)
            for db in unique_db:
                self.lsDb.insert(self.tk.END, self.get_icon_text("database", db))

            if unique_db:
                self.lsDb.selection_set(0)
                self.current_db = unique_db[0]

                if unique_db[0] in datasets_by_db:
                    self.lsDs.delete(0, self.tk.END)
                    for ds in datasets_by_db[unique_db[0]]:
                        self.lsDs.insert(self.tk.END, self.get_icon_text("data", ds))

            self.lsJSON.delete(0, self.tk.END)

            # Store search results
            self.search_datasets = datasets_by_db
            self.search_subjects = subject_map

            # Show summary
            total_datasets = sum(len(v) for v in datasets_by_db.values())
            self.txValue.delete("1.0", self.tk.END)
            self.txValue.insert(
                "1.0",
                f"Found {len(result)} results from {len(unique_db)} databases, {total_datasets} datasets",
            )

            # Hide search panel
            self.toggle_search()

        except Exception as e:
            self.messagebox.showerror("Search Error", str(e))

    def export_dataset(self):
        """Export the selected dataset"""

        # Use tracked current_db instead of relying on listbox selection
        if not self.current_db:
            self.messagebox.showwarning("Export", "Please select a database first")
            return

        # Try to get dataset from listbox selection, fall back to current_ds
        selection_ds = self.lsDs.curselection()
        if selection_ds:
            ds_name = self.lsDs.get(selection_ds[0])
            # Remove icon prefix
            ds_name = re.sub(r"^[^\s]+\s+", "", ds_name)
        elif self.current_ds:
            ds_name = self.current_ds
        else:
            self.messagebox.showwarning("Export", "Please select a dataset first")
            return

        db_name = self.current_db

        try:
            self.txValue.delete("1.0", self.tk.END)
            self.txValue.insert("1.0", f"Exporting {db_name}/{ds_name} ...")
            self.root.update()

            res = neuroj("export", db_name, ds_name)

            if res and "exportpath" in res:
                self.txValue.delete("1.0", self.tk.END)
                self.txValue.insert("1.0", f"Dataset exported to: {res['exportpath']}")
                self.messagebox.showinfo(
                    "Export Complete",
                    f"Dataset exported successfully to:\n{res['exportpath']}",
                )

        except Exception as e:
            self.messagebox.showerror("Export Error", str(e))
            import traceback

            traceback.print_exc()

    def loaddb(self):
        """Load database list"""
        try:
            dbs = neuroj("list")
            db_names = [dbitem["id"] for dbitem in dbs["database"]]

            self.lsDb.delete(0, self.tk.END)
            for name in db_names:
                self.lsDb.insert(self.tk.END, self.get_icon_text("database", name))

            # Clear search results
            self.search_datasets = {}
            self.search_subjects = {}

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load databases: {str(e)}")

    def loadds(self, event=None):
        """Load dataset list for selected database"""
        if time.time() - self.t0 < 0.01:
            return

        try:
            selection = self.lsDb.curselection()
            if not selection:
                return

            idx = selection[0]
            db_name = self.lsDb.get(idx)
            # Remove icon prefix
            db_name = re.sub(r"^[^\s]+\s+", "", db_name)
            self.current_db = db_name

            # Check if using search results
            if db_name in self.search_datasets:
                datasets = self.search_datasets[db_name]
                self.lsDs.delete(0, self.tk.END)
                for ds in datasets:
                    self.lsDs.insert(self.tk.END, self.get_icon_text("data", ds))
                self.lsJSON.delete(0, self.tk.END)
            else:
                dslist = neuroj("list", db_name)
                # Filter out system datasets starting with '_'
                datasets = [
                    dataset
                    for dataset in dslist["dataset"]
                    if not dataset["id"].startswith("_")
                ]

                self.lsDs.delete(0, self.tk.END)
                for dataset in datasets:
                    self.lsDs.insert(
                        self.tk.END, self.get_icon_text("data", dataset["id"])
                    )

            if self.lsDs.size() > 0:
                self.lsDs.selection_set(0)

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load datasets: {str(e)}")

        self.t0 = time.time()

    def loaddsdata(self, event=None):
        """Load data for selected dataset"""
        if time.time() - self.t0 < 0.01:
            return

        try:
            selection = self.lsDs.curselection()
            if not selection:
                return

            idx = selection[0]
            ds_name = self.lsDs.get(idx)
            # Remove icon prefix
            ds_name = re.sub(r"^[^\s]+\s+", "", ds_name)

            # Track current dataset
            self.current_ds = ds_name

            # Check if showing search subjects
            key = f"{self.current_db}/{ds_name}"
            if key in self.search_subjects and self.search_subjects[key]:
                subjects = self.search_subjects[key]
                self.lsJSON.delete(0, self.tk.END)
                for subj in subjects:
                    self.lsJSON.insert(
                        self.tk.END, self.get_icon_text("data", f"sub-{subj}")
                    )
                self.txValue.delete("1.0", self.tk.END)
                self.txValue.insert(
                    "1.0",
                    f"Showing {len(subjects)} subjects from search results for {key}",
                )
                self.datasets = None
                self.rootpath = ""
            else:
                # Show loading message
                self.txValue.delete("1.0", self.tk.END)
                self.txValue.insert("1.0", f"Loading {self.current_db}/{ds_name} ...")
                self.root.update()

                # Get dataset - neuroj('get') returns the parsed data
                self.datasets = neuroj("get", self.current_db, ds_name)

                # Display top-level keys
                if isinstance(self.datasets, dict) and len(self.datasets) > 0:
                    self.lsJSON.delete(0, self.tk.END)
                    for datakey in self.datasets.keys():
                        val = self.datasets.get(datakey)
                        icontype = self.detect_data_type(val, datakey)
                        self.lsJSON.insert(
                            self.tk.END, self.get_icon_text(icontype, datakey)
                        )

                    if self.lsJSON.size() > 0:
                        self.lsJSON.selection_set(0)

                    self.rootpath = ""
                    self.txValue.delete("1.0", self.tk.END)
                    self.txValue.insert(
                        "1.0",
                        f"Loaded {len(self.datasets)} keys from {self.current_db}/{ds_name}",
                    )
                elif self.datasets is None:
                    self.txValue.delete("1.0", self.tk.END)
                    self.txValue.insert("1.0", f"Failed to load dataset: returned None")
                else:
                    self.txValue.delete("1.0", self.tk.END)
                    self.txValue.insert(
                        "1.0",
                        f"Dataset type: {type(self.datasets)}\nContent: {str(self.datasets)[:500]}",
                    )

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            import traceback

            traceback.print_exc()

        self.t0 = time.time()

    def expandjsontree(self, event=None):
        """Expand JSON tree navigation"""
        if not isinstance(self.datasets, dict):
            return

        if time.time() - self.t0 < 0.01:
            return

        if len(self.rootpath) == 0:
            self.rootpath = "$"

        oldrootpath = self.rootpath

        try:
            selection = self.lsJSON.curselection()
            if not selection:
                return

            idx = selection[0]
            selected_key = self.lsJSON.get(idx)
            # Remove icon prefix
            selected_key = re.sub(r"^[^\s]+\s+", "", selected_key)

            if selected_key == "..":
                # Go back up
                if "." in self.rootpath:
                    self.rootpath = ".".join(re.split(r"(?<!\\)\.", self.rootpath)[:-1])
                else:
                    self.rootpath = "$"
            else:
                selected_key_escaped = selected_key.replace(".", "\\.")
                # Go deeper
                if self.rootpath:
                    self.rootpath += f".{selected_key_escaped}"
                else:
                    self.rootpath = "$." + selected_key_escaped

            # Navigate to the data
            current_data = jsonpath(self.datasets, self.rootpath)

            # Update display
            if isinstance(current_data, dict):
                keys = list(current_data.keys())
                self.lsJSON.delete(0, self.tk.END)

                if self.rootpath != "$":
                    self.lsJSON.insert(self.tk.END, self.get_icon_text("parent", ".."))

                for key in keys:
                    icontype = self.detect_data_type(current_data[key], key)
                    self.lsJSON.insert(self.tk.END, self.get_icon_text(icontype, key))

                if self.lsJSON.size() > 0:
                    self.lsJSON.selection_set(0)
            else:
                # Display value
                value_str = (
                    json.dumps(current_data, indent=2)
                    if not isinstance(current_data, str)
                    else str(current_data)
                )
                self.txValue.delete("1.0", self.tk.END)
                self.txValue.insert("1.0", value_str)
                self.rootpath = oldrootpath

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to expand tree: {str(e)}")

        self.t0 = time.time()


##====================================================================================
## Helper functions for HTTP requests
##====================================================================================


def _load_json_raw(url: str, **kwargs) -> str:
    """Load raw JSON string from URL"""

    if "limit" in kwargs:
        url += ("&" if (url.find("?") != -1) else "?") + f"limit={kwargs.get('limit')}"
    if "skip" in kwargs:
        url += ("&" if (url.find("?") != -1) else "?") + f"skip={kwargs.get('skip')}"

    req = urllib.request.Request(url)

    # Add headers from options
    weboptions = kwargs.get("weboptions", {})
    headers = weboptions.get("headers", {})
    for key, value in headers.items():
        req.add_header(key, value)

    # Add authentication if provided
    if "auth" in weboptions:
        username, password = weboptions["auth"]
        import base64

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ValueError(f"URL Error: {e.reason}")


def _load_json_header(url: str, **kwargs) -> Dict:
    """Load JSON header information"""
    req = urllib.request.Request(url)
    req.get_method = lambda: "HEAD"

    # Add headers from options
    weboptions = kwargs.get("weboptions", {})
    headers = weboptions.get("headers", {})
    for key, value in headers.items():
        req.add_header(key, value)

    # Add authentication if provided
    if "auth" in weboptions:
        username, password = weboptions["auth"]
        import base64

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req) as response:
            return dict(response.headers)
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ValueError(f"URL Error: {e.reason}")


def _web_write(url: str, data: str, options: Dict) -> Any:
    """Write data to web endpoint"""
    method = options.pop("method", "POST").upper()
    content_type = options.pop("content_type", "application/json")

    # Prepare request
    if isinstance(data, str):
        data = data.encode("utf-8")

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", content_type)
    req.get_method = lambda: method

    # Add additional headers
    headers = options.get("headers", {})
    for key, value in headers.items():
        req.add_header(key, value)

    # Add authentication if provided
    if "auth" in options:
        username, password = options["auth"]
        import base64

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ValueError(f"URL Error: {e.reason}")


def _web_upload(url: str, filepath: str, options: Dict) -> str:
    """Upload file to web endpoint"""
    import mimetypes

    # Read file content
    with open(filepath, "rb") as f:
        file_data = f.read()

    # Get content type
    content_type, _ = mimetypes.guess_type(filepath)
    if not content_type:
        content_type = "application/octet-stream"

    # Create multipart form data
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(filepath)

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode("utf-8")
    body += file_data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.get_method = lambda: "PUT"

    # Add authentication if provided
    if "auth" in options:
        username, password = options["auth"]
        import base64

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ValueError(f"URL Error: {e.reason}")
