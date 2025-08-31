"""
    Speed benchmark for saving/loading numpy arrays using various compression codecs
"""
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jdata as jd

print("jdata version:" + jd.__version__)

codecs = [
    "npy",
    "npz",
    "bjd",
    "zlib",
    "lzma",
    "lz4",
    "blosc2blosclz",
    "blosc2lz4",
    "blosc2lz4hc",
    "blosc2zlib",
    "blosc2zstd",
]
nthread = 8


def benchmark(codec, x):
    t0 = time.time()
    ext = suffix
    if codec == "npy":
        ext = "." + codec
        np.save("matrix_" + codec + ext, x)
    elif codec == "npz":
        ext = "." + codec
        np.savez_compressed("matrix_" + codec + ext, x)
    elif codec == "bjd":
        ext = "." + codec
        jd.save(x, "matrix_" + codec + ext, {"encode": False})
    else:
        jd.save(x, "matrix_" + codec + ext, {"compression": codec, "nthread": nthread})
    dt = time.time() - t0  # saving time
    res = {"codec": codec, "save": dt}
    if codec == "npy":
        y = np.load("matrix_" + codec + ext)
    elif codec == "npz":
        y = np.load("matrix_" + codec + ext)["arr_0"]
    else:
        y = jd.load("matrix_" + codec + ext, {"nthread": nthread})  # loading
    res["sum"] = y.sum()
    res["load"] = time.time() - t0 - dt  # loading time
    res["size"] = os.path.getsize("matrix_" + codec + ext)
    print(res)
    return res


## a highly compressible matrix
x = np.eye(10000)

## a less compressible random matrix
# np.random.seed(0)
# x = np.random.rand(2000,2000)

print("\n- Testing binary JSON (BJData) files (.jdb) ...")

suffix = ".jdb"
res = list(map(benchmark, codecs, [x] * len(codecs)))
# print(np.array(res))

print("\n- Testing text-based JSON files (.jdt) ...")

suffix = ".jdt"
res = list(map(benchmark, codecs, [x] * len(codecs)))
# print(np.array(res))
