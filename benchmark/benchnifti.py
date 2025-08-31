import time
import os
import glob
import urllib.request
import zipfile
import tempfile
import shutil

import nibabel as nib
import numpy as np
import jdata as jd

tempdir = tempfile.mkdtemp()

url = "https://github.com/neurolabusc/niivue-images/archive/refs/heads/main.zip"
fname = os.path.join(tempdir, "niivue-images.zip")
urllib.request.urlretrieve(url, fname)

with zipfile.ZipFile(fname, "r") as zip_ref:
    zip_ref.extractall(tempdir)

niifiles = glob.glob(os.path.join(tempdir, "niivue-images-main/", "*.nii.gz"))

for ff in niifiles:
    # benchmark loading time from nib.load()
    t0 = time.time()
    img = nib.load(ff)
    data = np.asarray(img.dataobj)
    try:
        s1 = np.sum(data).item()
    except:
        s1 = -1
    dt1 = time.time() - t0

    # benchmark loading time from jd.loadnifti()
    t1 = time.time()
    nii = jd.loadnifti(ff)
    s2 = np.sum(nii["NIFTIData"]).item()
    dt2 = time.time() - t1
    jd.show(
        {
            "file": ff,
            "nib": [list(data.shape), data.dtype.str, s1],
            "jd": [list(nii["NIFTIData"].shape), nii["NIFTIData"].dtype.str, s2],
            "nibtime": dt1,
            "jdtime": dt2,
            "speedup": dt1 / dt2,
        }
    )

try:
    shutil.rmtree(tempdir)
except OSError as e:
    print(f"unable to delete the temporary folder: {e}")
