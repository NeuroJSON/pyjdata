"""@package docstring
JSONPath implementation ported from the jsonpath MATLAB function in JSONLab

Copyright (c) 2019-2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "jsonpath",
]

##====================================================================================
## dependent libraries
##====================================================================================


import re
import json
import copy


def jsonpath(root, jpath, opt={}):

    obj = root
    jpath = re.sub(r"([^.\]])(\[[-0-9:\*]+\])", r"\1.\2", jpath)
    jpath = re.sub(r"\[[\'\"]*([^]\'\"]+)[\'\"]*\]", r".[\1]", jpath)
    jpath = re.sub(r"\\.", "_0x2E_", jpath)
    while re.search(r"(\[[\'\"]*[^]\'\"]+)\.(?=[^]\'\"]+[\'\"]*\])", jpath):
        jpath = re.sub(
            r"(\[[\'\"]*[^]\'\"]+)\.(?=[^]\'\"]+[\'\"]*\])", r"\1_0x2E_", jpath
        )

    paths = re.findall(r"(\.{0,2}[^.]+)", jpath)
    paths = [re.sub("_0x2E_", ".", x) for x in paths]
    if paths and paths[0] == "$":
        paths.pop(0)

    for i, path in enumerate(paths):
        obj, isfound = getonelevel(obj, paths, i, opt)
        if not isfound:
            return None
    return obj


def getonelevel(input_data, paths, pathid, opt):

    opt.setdefault("inplace", False)

    pathname = paths[pathid]
    if isinstance(pathname, list):
        pathname = pathname[0]
    deepscan = bool(re.search(r"^\.\.", pathname))
    origpath = pathname
    pathname = re.sub(r"^\.+", "", pathname)
    obj = None
    isfound = False

    if pathname == "$":
        obj = input_data
    elif re.match(r"\$\d+", pathname):
        obj = input_data[int(pathname[2:]) + 1]
    elif re.match(r"^\[[\-0-9\*:]+\]$", pathname) or isinstance(
        input_data, (list, tuple, frozenset)
    ):
        arraystr = pathname[1:-1]
        arrayrange = {"start": None, "end": None}

        if ":" in arraystr:
            match = re.search(r"(?P<start>-*\d*):(?P<end>-*\d*)", arraystr)
            if match:
                arrayrange["start"] = (
                    int(match.group("start")) if match.group("start") else None
                )
                arrayrange["end"] = (
                    int(match.group("end")) if match.group("end") else None
                )

                if arrayrange["start"] is not None:
                    if arrayrange["start"] < 0:
                        arrayrange["start"] = len(input_data) + arrayrange["start"] + 1
                    else:
                        arrayrange["start"] += 1
                else:
                    arrayrange["start"] = 1

                if arrayrange["end"] is not None:
                    if arrayrange["end"] < 0:
                        arrayrange["end"] = len(input_data) + arrayrange["end"] + 1
                    else:
                        arrayrange["end"] += 1
                else:
                    arrayrange["end"] = len(input_data)
        elif re.match(r"^[-0-9:]+$", arraystr):
            firstidx = int(arraystr)
            if firstidx < 0:
                firstidx = len(input_data) + firstidx + 1
            else:
                firstidx += 1
            arrayrange["start"] = arrayrange["end"] = firstidx
        elif re.match(r"^\*$", arraystr):
            arrayrange = {"start": 1, "end": len(input_data)}

        if (
            "arrayrange" in locals()
            and arrayrange["start"] is not None
            and arrayrange["end"] is not None
        ):
            obj = input_data[arrayrange["start"] - 1 : arrayrange["end"]]
        else:
            arrayrange = {"start": 1, "end": len(input_data)}

        if not obj and isinstance(input_data, list):
            input_data = input_data[arrayrange["start"] - 1 : arrayrange["end"]]
            searchkey = ".." + pathname if deepscan else origpath
            newobj = []
            for idx, item in enumerate(input_data):
                val, isfound = getonelevel(
                    item, paths[:pathid] + [searchkey], pathid, opt
                )
                if isfound:
                    if isinstance(val, list):
                        if len(val) > 1:
                            newobj.extend(val)
                        else:
                            newobj.append(val)
                    else:
                        newobj.append(val)
            if newobj:
                obj = newobj
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]

    elif isinstance(input_data, dict):
        pathname = re.sub(r"^\[(.*)\]$", r"\1", pathname)
        stpath = pathname

        if stpath in input_data:
            obj = [input_data[stpath]]

        if obj is None or deepscan:
            items = input_data.keys()

            for idx in items:
                val, isfound = getonelevel(
                    input_data[idx], paths[:pathid] + [[".." + pathname]], pathid, opt
                )
                if isfound:
                    obj = obj or []
                    if isinstance(val, list):
                        if len(val) > 1:
                            obj.extend(val)
                        else:
                            obj.append(val)
                    else:
                        obj.append(val)

            if obj and len(obj) == 1:
                obj = obj[0]

        if isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]

    elif not deepscan:
        raise ValueError(
            f'json path segment "{pathname}" can not be found in the input_data object'
        )

    if obj is None:
        isfound = False
        obj = []
    else:
        isfound = True

    return (copy.deepcopy(obj), isfound) if opt["inplace"] else (obj, isfound)
