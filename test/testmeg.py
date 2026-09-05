"""Round-trip tests for the MEG/iEEG container wrappers: CTF, FIFF, MEF3.

Every test builds its own file, so the expected bytes are known exactly and
nothing here depends on a corpus being present.
"""

import hashlib
import os
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jdata.jctf import ctf2jctf, jctf2ctf, ctfinfo
from jdata.jfiff import fiff2jfiff, jfiff2fiff, fiffinfo
from jdata.jmef3 import mef32jmef3, jmef32mef3, mef3info
from jdata.njencode import ENCODABLE, CONTAINER_EXT, container_digest, encoder_for

RES4_MAGIC = b"MEG42RS\x00"
MEG4_MAGIC = b"MEG41CP\x00"


def write_ctf(dspath, nchan=4, nsamp=8, ntrials=3, srate=1200.0, extra=True):
    """A minimal but structurally valid CTF .ds directory."""
    os.makedirs(dspath, exist_ok=True)
    stem = os.path.basename(dspath)[:-3]

    raw = bytearray(1847 + nchan * 32 + nchan * 16)
    raw[0:8] = RES4_MAGIC
    raw[8:8 + 7] = b"testapp"
    struct.pack_into(">h", raw, 776, 0)
    raw[778:778 + 5] = b"09:41"
    raw[1033:1033 + 11] = b"01-Jan-2020"
    struct.pack_into(">i", raw, 1288, nsamp)
    struct.pack_into(">h", raw, 1292, nchan)
    struct.pack_into(">d", raw, 1296, srate)
    struct.pack_into(">d", raw, 1304, ntrials * nsamp / srate)
    struct.pack_into(">h", raw, 1312, ntrials)
    for k in range(nchan):
        name = ("MLC%02d" % k).encode()
        raw[1847 + 32 * k : 1847 + 32 * k + len(name)] = name
    with open(os.path.join(dspath, stem + ".res4"), "wb") as fid:
        fid.write(bytes(raw))

    samples = np.arange(ntrials * nchan * nsamp, dtype=np.int32).reshape(ntrials, nchan, nsamp)
    with open(os.path.join(dspath, stem + ".meg4"), "wb") as fid:
        fid.write(MEG4_MAGIC)
        fid.write(samples.astype(">i4").tobytes())

    if extra:
        with open(os.path.join(dspath, "BadChannels"), "wb") as fid:
            fid.write(b"MLC01\nMLC02\n")
        with open(os.path.join(dspath, stem + ".hc"), "wb") as fid:
            fid.write(b"standard nasion coil position relative to dewar (cm):\n\tx = 1.5\n")
        with open(os.path.join(dspath, "MarkerFile.mrk"), "wb") as fid:
            fid.write(
                b"PATH OF DATASET:\n/tmp/x.ds\n\nNUMBER OF MARKERS:\n1\n\n"
                b"CLASSGROUPID:\n3\nNAME:\ntrig\nCOMMENT:\nnone\nCOLOR:\nred\n"
                b"EDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n2\n"
                b"LIST OF SAMPLES:\nTRIAL NUMBER\t\tTIME FROM SYNC POINT\n"
                b"                  +0\t\t\t\t          +0.5000\n"
                b"                  +1\t\t\t\t          +1.2500\n"
            )
    return samples


def write_fiff(path, ntag=6, bulkbytes=70000, kinds=None):
    """A tag stream with a mix of small tags and one oversized payload."""
    out = bytearray()
    payloads = []
    for k in range(ntag):
        size = bulkbytes if k == ntag - 2 else 12
        body = bytes((k * 7 + i) % 251 for i in range(min(size, 4096)))
        body = (body * (size // len(body) + 1))[:size]
        payloads.append(body)
        kind = kinds[k] if kinds else 100 + k
        out += struct.pack(">4i", kind, 3, size, 0)
        out += body
    with open(path, "wb") as fid:
        fid.write(bytes(out))
    return payloads


def write_mef3(mefd, channels=("LAA1", "LAA2"), nblocks=3, red=64):
    """A .mefd tree with a valid 1024-byte universal header on every file."""

    def uh(kind, chan, entries):
        raw = bytearray(1024)
        struct.pack_into("<I", raw, 0, 0xDEADBEEF)
        struct.pack_into("<I", raw, 4, 0x12345678)
        raw[8 : 8 + len(kind)] = kind
        raw[13] = 3
        raw[14] = 0
        raw[15] = 1
        struct.pack_into("<q", raw, 16, 1000)
        struct.pack_into("<q", raw, 24, 2000)
        struct.pack_into("<q", raw, 32, entries)
        struct.pack_into("<q", raw, 40, 4096)
        struct.pack_into("<i", raw, 48, 0)
        raw[52 : 52 + len(chan)] = chan.encode()
        raw[308 : 308 + 4] = b"sess"
        raw[564 : 564 + 11] = b"not_entered"
        # protected region: non-zero, to catch a rebuild that drops it
        raw[900:916] = bytes(range(16))
        return bytes(raw)

    for chan in channels:
        seg = os.path.join(mefd, chan + ".timd", chan + "-000000.segd")
        os.makedirs(seg, exist_ok=True)
        stem = chan + "-000000"

        met = bytearray(uh(b"tmet", chan, 1) + bytes(15360))
        struct.pack_into("<d", met, 8720, 512.0)
        struct.pack_into("<d", met, 8752, 60.0)
        with open(os.path.join(seg, stem + ".tmet"), "wb") as fid:
            fid.write(bytes(met))

        idx = bytearray(uh(b"tidx", chan, nblocks))
        for b in range(nblocks):
            rec = bytearray(56)
            struct.pack_into("<q", rec, 0, 1024 + b * red)
            struct.pack_into("<q", rec, 8, 1000 + b)
            struct.pack_into("<q", rec, 16, b * 16)
            struct.pack_into("<I", rec, 24, 16)
            struct.pack_into("<I", rec, 28, red)
            idx += rec
        with open(os.path.join(seg, stem + ".tidx"), "wb") as fid:
            fid.write(bytes(idx))

        with open(os.path.join(seg, stem + ".tdat"), "wb") as fid:
            fid.write(uh(b"tdat", chan, nblocks))
            fid.write(bytes((i * 3) % 256 for i in range(nblocks * red)))


def digests(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            with open(full, "rb") as fid:
                out[os.path.relpath(full, root)] = hashlib.sha256(fid.read()).hexdigest()
    return out


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCTF(TempCase):
    def build(self, **kw):
        ds = os.path.join(self.tmp, "sub-01_task-x_meg.ds")
        return ds, write_ctf(ds, **kw)

    def test_header_fields(self):
        ds, _ = self.build()
        head = ctf2jctf(ds)["CTFData"]["res4"]
        self.assertEqual(head["no_channels"], 4)
        self.assertEqual(head["no_samples"], 8)
        self.assertEqual(head["no_trials"], 3)
        self.assertEqual(head["sample_rate"], 1200.0)
        self.assertEqual(head["data_date"], "01-Jan-2020")
        self.assertEqual(head["chanName"][:2], ["MLC00", "MLC01"])

    def test_epoch_time_invariant(self):
        """epoch_time == no_trials * no_samples / sample_rate, the check that
        pinned the no_trials offset against real recordings."""
        ds, _ = self.build()
        h = ctf2jctf(ds)["CTFData"]["res4"]
        self.assertAlmostEqual(h["epoch_time"], h["no_trials"] * h["no_samples"] / h["sample_rate"])

    def test_samples_exact(self):
        ds, samples = self.build()
        got = ctf2jctf(ds)["CTFData"]["meg4"]
        self.assertEqual(got.shape, (3, 4, 8))
        self.assertTrue(np.array_equal(got, samples))

    def test_sample_layout_is_channel_major(self):
        """Channel 0 of trial 0 must be the first nsamp values, not a stride."""
        ds, samples = self.build()
        got = ctf2jctf(ds)["CTFData"]["meg4"]
        self.assertTrue(np.array_equal(got[0, 0], np.arange(8)))

    def test_roundtrip_byte_exact(self):
        ds, _ = self.build()
        before = digests(ds)
        out = os.path.join(self.tmp, "rebuilt.ds")
        jctf2ctf(ctf2jctf(ds), out)
        self.assertEqual(digests(out), before)

    def test_res4_kept_verbatim(self):
        ds, _ = self.build()
        with open(os.path.join(ds, "sub-01_task-x_meg.res4"), "rb") as fid:
            raw = fid.read()
        self.assertEqual(bytes(ctf2jctf(ds)["CTFData"]["res4"]["_ByteStream_"]), raw)

    def test_markerfile_parsed_with_its_own_names(self):
        ds, _ = self.build()
        mrk = ctf2jctf(ds)["CTFData"]["MarkerFile.mrk"]
        self.assertEqual(mrk["NUMBER OF MARKERS"], 1)
        self.assertEqual(mrk["markers"][0]["NAME"], "trig")
        self.assertEqual(mrk["markers"][0]["samples"], [[0, 0.5], [1, 1.25]])

    def test_hc_kept_as_bytes(self):
        """.hc names its contents in prose, so there is no key set to adopt."""
        ds, _ = self.build()
        node = ctf2jctf(ds)["CTFData"]["hc"]
        self.assertEqual(list(node.keys()), ["_ByteStream_"])

    def test_badchannels_listed(self):
        ds, _ = self.build()
        self.assertEqual(ctf2jctf(ds)["CTFData"]["BadChannels"]["channels"], ["MLC01", "MLC02"])

    def test_info_summary(self):
        ds, _ = self.build()
        info = ctfinfo(ds)
        self.assertEqual(info["Format"], "CTF")
        self.assertEqual(info["NumberOfChannels"], 4)
        self.assertEqual(info["SplitParts"], 1)

    def test_rejects_bad_magic(self):
        ds, _ = self.build()
        target = os.path.join(ds, "sub-01_task-x_meg.res4")
        with open(target, "r+b") as fid:
            fid.write(b"NOTCTF\x00\x00")
        self.assertRaises(ValueError, ctf2jctf, ds)


class TestFIFF(TempCase):
    def build(self, **kw):
        path = os.path.join(self.tmp, "sub-01_meg.fif")
        return path, write_fiff(path, **kw)

    def test_tag_directory(self):
        path, payloads = self.build()
        d = fiff2jfiff(path)["FIFFData"]
        self.assertEqual(d["TagDirectory"].shape, (6, 4))
        self.assertEqual(list(d["TagDirectory"][0]), [100, 3, 12, 0])

    def test_bulk_lifted_by_size_not_kind(self):
        """Bulk selection must not depend on the unverified constants table."""
        path, payloads = self.build()
        d = fiff2jfiff(path)["FIFFData"]
        self.assertEqual(len(d["TagData"]), 70000)
        self.assertEqual(d["Tags"][4]["DataOffset"], 0)
        self.assertNotIn("DataOffset", d["Tags"][0])

    def test_kind_is_authoritative_name_is_advisory(self):
        """A kind absent from the advisory table must still round-trip."""
        path, _ = self.build(kinds=[100, 101, 103, 104, 105, 31337])
        tags = fiff2jfiff(path)["FIFFData"]["Tags"]
        self.assertEqual(tags[0]["Kind"], 100)
        self.assertEqual(tags[0]["KindName"], "FIFF_FILE_ID")
        self.assertEqual(tags[5]["Kind"], 31337)
        self.assertNotIn("KindName", tags[5])
        out = os.path.join(self.tmp, "unknown.fif")
        with open(path, "rb") as fid:
            want = hashlib.sha256(fid.read()).hexdigest()
        self.assertEqual(jfiff2fiff(fiff2jfiff(path), out), want)

    def test_roundtrip_byte_exact(self):
        path, _ = self.build()
        with open(path, "rb") as fid:
            want = hashlib.sha256(fid.read()).hexdigest()
        out = os.path.join(self.tmp, "rebuilt.fif")
        self.assertEqual(jfiff2fiff(fiff2jfiff(path), out), want)

    def test_info_summary(self):
        path, _ = self.build()
        info = fiffinfo(path)
        self.assertEqual(info["NumberOfTags"], 6)
        self.assertEqual(info["NumberOfBulkTags"], 1)
        self.assertEqual(info["Endian"], "big")

    def test_truncated_stream_stops_cleanly(self):
        path, _ = self.build()
        with open(path, "rb") as fid:
            raw = fid.read()
        with open(path, "wb") as fid:
            fid.write(raw[: len(raw) - 40])
        self.assertLess(len(fiff2jfiff(path)["FIFFData"]["Tags"]), 6)


class TestMEF3(TempCase):
    def build(self, **kw):
        mefd = os.path.join(self.tmp, "sub-01_ieeg.mefd")
        write_mef3(mefd, **kw)
        return mefd

    def test_tree_shape(self):
        mefd = self.build()
        ch = mef32jmef3(mefd)["MEF3Data"]["time_series_channels"]
        self.assertEqual(sorted(ch), ["LAA1", "LAA2"])
        self.assertEqual(len(ch["LAA1"]["segments"]), 1)

    def test_universal_header_fields(self):
        mefd = self.build()
        seg = mef32jmef3(mefd)["MEF3Data"]["time_series_channels"]["LAA1"]["segments"][0]
        uh = seg["metadata"]["universal_header"]
        self.assertEqual(uh["file_type_string"], "tmet")
        self.assertEqual(uh["mef_version_major"], 3)
        self.assertEqual(uh["byte_order_code"], 1)
        self.assertEqual(uh["channel_name"], "LAA1")
        self.assertFalse(uh["level_1_encrypted"])

    def test_section2_native_names(self):
        mefd = self.build()
        seg = mef32jmef3(mefd)["MEF3Data"]["time_series_channels"]["LAA1"]["segments"][0]
        s2 = seg["metadata"]["section_2"]
        self.assertEqual(s2["sampling_frequency"], 512.0)
        self.assertEqual(s2["AC_line_frequency"], 60.0)

    def test_index_record_size(self):
        mefd = self.build(nblocks=5)
        seg = mef32jmef3(mefd)["MEF3Data"]["time_series_channels"]["LAA1"]["segments"][0]
        self.assertEqual(seg["time_series_indices"]["records"].shape, (5, 56))

    def test_payload_kept_as_red(self):
        mefd = self.build(nblocks=3, red=64)
        j = mef32jmef3(mefd)
        seg = j["MEF3Data"]["time_series_channels"]["LAA1"]["segments"][0]
        self.assertEqual(len(seg["time_series_data"]["RED"]), 3 * 64)
        self.assertEqual(j["MEF3Source"]["PayloadMode"], "verbatim")
        self.assertEqual(j["MEF3Source"]["Compression"], "RED")

    def test_roundtrip_byte_exact(self):
        mefd = self.build()
        before = digests(mefd)
        out = os.path.join(self.tmp, "rebuilt.mefd")
        jmef32mef3(mef32jmef3(mefd), out)
        self.assertEqual(digests(out), before)

    def test_protected_region_survives(self):
        """Bytes 900-1024 are unparsed; a rebuild from parsed fields alone
        would silently drop them."""
        mefd = self.build()
        out = os.path.join(self.tmp, "rebuilt.mefd")
        jmef32mef3(mef32jmef3(mefd), out)
        rel = "LAA1.timd/LAA1-000000.segd/LAA1-000000.tdat"
        with open(os.path.join(out, rel), "rb") as fid:
            raw = fid.read(1024)
        self.assertEqual(raw[900:916], bytes(range(16)))

    def test_encryption_detected(self):
        mefd = self.build()
        target = os.path.join(mefd, "LAA1.timd", "LAA1-000000.segd", "LAA1-000000.tmet")
        with open(target, "r+b") as fid:
            fid.seek(868)
            fid.write(b"\x01" * 16)
        self.assertTrue(mef3info(mefd)["Encrypted"])
        self.assertTrue(mef32jmef3(mefd)["MEF3Source"]["Encrypted"])

    def test_info_summary(self):
        mefd = self.build()
        info = mef3info(mefd)
        self.assertEqual(info["Format"], "MEF3")
        self.assertEqual(info["MEFVersion"], "3.0")
        self.assertEqual(info["NumberOfChannels"], 2)
        self.assertEqual(info["SamplingFrequency"], 512.0)


class TestDispatch(TempCase):
    def test_attachment_names_are_binary_spellings(self):
        """Payloads are BJData, so the suffix must be b*, not j*."""
        for ext in (".ds", ".mefd", ".fif", ".edf", ".bdf", ".vhdr", ".set"):
            self.assertTrue(encoder_for(ext)[0].startswith(".b"), ext)

    def test_encodable_entries(self):
        self.assertEqual(encoder_for(".ds"), (".bmeg", ("CTFData",)))
        self.assertEqual(encoder_for(".mefd"), (".bmef", ("MEF3Data",)))
        self.assertEqual(encoder_for(".fif"), (".bfif", ("FIFFData",)))

    def test_container_ext(self):
        self.assertEqual(sorted(CONTAINER_EXT), [".ds", ".mefd"])

    def test_container_digest_is_stable_and_content_sensitive(self):
        a = os.path.join(self.tmp, "a.ds")
        b = os.path.join(self.tmp, "b.ds")
        write_ctf(a)
        write_ctf(b)
        # different container names, identical member content and relpaths?
        # relpaths embed the stem, so digests differ -- what must hold is that
        # a container digests the same twice, and changes when a byte changes.
        first, n = container_digest(a)
        self.assertEqual(first, container_digest(a)[0])
        self.assertEqual(n, len(digests(a)))
        with open(os.path.join(a, "BadChannels"), "ab") as fid:
            fid.write(b"MLC03\n")
        self.assertNotEqual(first, container_digest(a)[0])

    def test_walk_treats_containers_as_units(self):
        from jdata.njbids import _walk

        root = os.path.join(self.tmp, "sub-01", "meg")
        os.makedirs(root)
        write_ctf(os.path.join(root, "sub-01_task-x_meg.ds"))
        rels = [f.relpath for f in _walk(self.tmp)]
        self.assertIn("sub-01/meg/sub-01_task-x_meg.ds", rels)
        self.assertFalse([r for r in rels if ".ds/" in r])

    def test_walk_container_size_is_sum_of_members(self):
        from jdata.njbids import _walk

        ds = os.path.join(self.tmp, "sub-01_task-x_meg.ds")
        write_ctf(ds)
        want = sum(os.path.getsize(os.path.join(ds, n)) for n in os.listdir(ds))
        entry = [f for f in _walk(self.tmp) if f.relpath.endswith(".ds")][0]
        self.assertEqual(entry.size, want)
        self.assertTrue(entry.present)


if __name__ == "__main__":
    unittest.main()
