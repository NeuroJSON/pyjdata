import jdata as jd
import numpy as np
import time
import os

print("jdata version:" + jd.__version__)

codecs = ["zlib", "lzma", "lz4", "blosc2blosclz", "blosc2lz4", "blosc2lz4hc", "blosc2zlib", "blosc2zstd"]

def benchmark(codec, x):
    t0 = time.time()
    jd.save(x, "matrix_" + codec + suffix, {"compression": codec, "nthread": 8})
    dt = time.time() - t0  # saving time
    res = {"codec": codec, "save": dt}
    y = jd.load("matrix_" + codec + suffix, {"nthread": 8})  # loading
    res["load"] = time.time() - t0 - dt  # loading time
    res["size"] = os.path.getsize("matrix_" + codec + suffix)
    res["sum"] = y.sum()
    print(res)
    return res


x = np.eye(10000)
suffix = '.jdb'
res = list(map(benchmark, codecs, [x] * len(codecs)))
# print(np.array(res))

suffix = '.jdt'
res = list(map(benchmark, codecs, [x] * len(codecs)))
# print(np.array(res))
