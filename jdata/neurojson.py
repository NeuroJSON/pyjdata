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
            jsonstring = _load_json_raw(restapi, **kwargs)
            res = json.loads(jsonstring)
        else:
            restapi = f"{serverurl}{db}/{ds}/{file}"
            res, jsonstring = jdlink(restapi)

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


class neurojgui:
    """
    GUI class for NeuroJSON browser
    """

    def __init__(self):
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox

        self.root = self.tk.Tk()
        self.root.title("NeuroJSON.io Dataset Browser")
        self.root.geometry("800x600")

        # Create frames
        self.create_widgets()
        self.t0 = time.time()

    def create_widgets(self):
        """Create GUI widgets"""

        # Create toolbar frame
        toolbar = self.tk.Frame(self.root)
        toolbar.pack(side=self.tk.TOP, fill=self.tk.X)

        # Load database button
        btn_load_db = self.tk.Button(
            toolbar, text="List Databases", command=self.loaddb
        )
        btn_load_db.pack(side=self.tk.LEFT, padx=5, pady=5)

        # Create main frame
        main_frame = self.tk.Frame(self.root)
        main_frame.pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)

        # Database listbox
        db_frame = self.tk.Frame(main_frame)
        db_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH)

        self.tk.Label(db_frame, text="Database").pack()
        self.lsDb = self.tk.Listbox(db_frame, width=20)
        self.lsDb.pack(fill=self.tk.BOTH, expand=True)
        self.lsDb.bind("<Double-Button-1>", self.loadds)
        self.lsDb.bind("<Return>", self.loadds)

        # Dataset listbox
        ds_frame = self.tk.Frame(main_frame)
        ds_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH)

        self.tk.Label(ds_frame, text="Dataset").pack()
        self.lsDs = self.tk.Listbox(ds_frame, width=25)
        self.lsDs.pack(fill=self.tk.BOTH, expand=True)
        self.lsDs.bind("<Double-Button-1>", self.loaddsdata)
        self.lsDs.bind("<Return>", self.loaddsdata)

        # JSON tree listbox
        json_frame = self.tk.Frame(main_frame)
        json_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        self.tk.Label(json_frame, text="Data").pack()
        self.lsJSON = self.tk.Listbox(json_frame)
        self.lsJSON.pack(fill=self.tk.BOTH, expand=True)
        self.lsJSON.bind("<Double-Button-1>", self.expandjsontree)
        self.lsJSON.bind("<Return>", self.expandjsontree)

        # Value text area
        value_frame = self.tk.Frame(main_frame)
        value_frame.pack(side=self.tk.RIGHT, fill=self.tk.BOTH)

        self.tk.Label(value_frame, text="Value").pack()
        self.txValue = self.tk.Text(value_frame, width=30)
        self.txValue.pack(fill=self.tk.BOTH, expand=True)

        # Initialize data storage
        self.current_db = ""
        self.datasets = None
        self.rootpath = ""

    def run(self):
        """Start the GUI"""
        self.root.mainloop()

    def loaddb(self):
        """Load database list"""
        try:
            dbs = neuroj("list")
            db_names = [dbitem["id"] for dbitem in dbs["database"]]

            self.lsDb.delete(0, self.tk.END)
            for name in db_names:
                self.lsDb.insert(self.tk.END, name)

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
            self.current_db = db_name

            dslist = neuroj("list", db_name)
            # Filter out system datasets starting with '_'
            datasets = [
                dataset
                for dataset in dslist["dataset"]
                if not dataset["id"].startswith("_")
            ]

            self.lsDs.delete(0, self.tk.END)
            for dataset in datasets:
                self.lsDs.insert(self.tk.END, dataset["id"])

            if datasets:
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

            self.datasets = neuroj("get", self.current_db, ds_name)

            # Display top-level keys
            if isinstance(self.datasets, dict):
                keys = self.datasets.keys()
                self.lsJSON.delete(0, self.tk.END)
                for key in keys:
                    self.lsJSON.insert(self.tk.END, key)

                if keys:
                    self.lsJSON.selection_set(0)

                self.rootpath = ""

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

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

            if selected_key == "..":
                # Go back up
                if "." in self.rootpath:
                    self.rootpath = ".".join(
                        re.split("(?<!\\\\)\.", self.rootpath)[:-1]
                    )
                else:
                    self.rootpath = "$"
            else:
                selected_key = selected_key.replace(".", "\\.")
                # Go deeper
                if self.rootpath:
                    self.rootpath += f".{selected_key}"
                else:
                    self.rootpath = "$." + selected_key

            # Navigate to the data
            current_data = jsonpath(self.datasets, self.rootpath)

            # Update display
            if isinstance(current_data, dict):
                keys = current_data.keys()
                if self.rootpath != "$":
                    keys = [".."] + list(keys)

                self.lsJSON.delete(0, self.tk.END)
                for key in keys:
                    self.lsJSON.insert(self.tk.END, key)

                if keys:
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
