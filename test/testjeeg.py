"""Tests for jdata.jeeg -- electrophysiology recordings as JData.

The recordings are built here rather than fetched, so the expected samples are
known exactly and the tests say something about correctness rather than just
about not raising.

To run:

    python3 -m unittest test.testjeeg
"""

import os
import shutil
import hashlib
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jdata.jeeg import (bv2jeeg, edf2jeeg, eeg2jeeg, eeglab2jeeg, eeginfo,
                        jeeg2bv, jeeg2edf, jeeg2eeglab)


def write_edf(path, nsig=2, nrec=3, persig=4, srate_dur=1.0, bdf=False, signals=None):
    """A minimal but standards-shaped EDF/BDF file.

    Returns the digital samples written, as ``nsig x (nrec * persig)``.
    """
    width = 3 if bdf else 2
    dmin, dmax = (-8388608, 8388607) if bdf else (-32768, 32767)
    pmin, pmax = -1000.0, 1000.0

    if signals is None:
        rng = np.arange(nsig * nrec * persig, dtype=np.int64)
        signals = ((rng % 201) - 100).reshape(nsig, nrec * persig).astype(np.int64)

    def pad(text, n):
        return ("%-*s" % (n, text))[:n].encode("latin-1")

    head = bytearray()
    head += (b"\xffBIOSEMI" if bdf else pad("0", 8))
    head += pad("patient x", 80)
    head += pad("recording y", 80)
    head += pad("01.02.03", 8)
    head += pad("04.05.06", 8)
    head += pad(str(256 + 256 * nsig), 8)
    head += pad("24BIT" if bdf else "", 44)
    head += pad(str(nrec), 8)
    head += pad("%g" % srate_dur, 8)
    head += pad(str(nsig), 4)
    assert len(head) == 256

    def block(values, n):
        return b"".join(pad(v, n) for v in values)

    head += block(["ch%d" % (i + 1) for i in range(nsig)], 16)
    head += block(["AgAgCl"] * nsig, 80)
    head += block(["uV"] * nsig, 8)
    head += block(["%g" % pmin] * nsig, 8)
    head += block(["%g" % pmax] * nsig, 8)
    head += block([str(dmin)] * nsig, 8)
    head += block([str(dmax)] * nsig, 8)
    head += block(["HP:0.1Hz"] * nsig, 80)
    head += block([str(persig)] * nsig, 8)
    head += block([""] * nsig, 32)

    body = bytearray()

    for r in range(nrec):
        for i in range(nsig):
            for v in signals[i, r * persig : (r + 1) * persig]:
                iv = int(v)

                if bdf:
                    body += (iv & 0xFFFFFF).to_bytes(3, "little")
                else:
                    body += struct.pack("<h", iv)

    with open(path, "wb") as fid:
        fid.write(bytes(head) + bytes(body))

    return signals


def write_brainvision(path, nchan=3, nsamp=50, orientation="MULTIPLEXED"):
    """A BrainVision .vhdr plus its float32 .eeg. Returns the samples."""
    stem = os.path.splitext(os.path.basename(path))[0]
    data = (np.arange(nchan * nsamp, dtype=np.float32).reshape(nchan, nsamp) / 7.0).astype(
        "<f4"
    )
    raw = data.T.copy() if orientation == "MULTIPLEXED" else data.copy()

    with open(os.path.join(os.path.dirname(path), stem + ".eeg"), "wb") as fid:
        fid.write(raw.tobytes(order="C"))

    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "",
        "[Common Infos]",
        "DataFile=%s.eeg" % stem,
        "MarkerFile=%s.vmrk" % stem,
        "DataFormat=BINARY",
        "DataOrientation=%s" % orientation,
        "NumberOfChannels=%d" % nchan,
        "SamplingInterval=500",
        "",
        "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32",
        "",
        "[Channel Infos]",
    ]
    lines += ["Ch%d=E%d,,1.0,uV" % (i + 1, i + 1) for i in range(nchan)]

    with open(path, "w") as fid:
        fid.write("\n".join(lines) + "\n")

    return data


class TestEDF(unittest.TestCase):
    def test_samples_round_trip_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.edf")
            expect = write_edf(path, nsig=2, nrec=3, persig=4)
            got = edf2jeeg(path)
            np.testing.assert_array_equal(got["EEGData"], expect)
            self.assertEqual(got["EEGHeader"]["Format"], "EDF")
            self.assertEqual(got["EEGHeader"]["NumberOfSignals"], 2)

    def test_records_are_deinterleaved_per_channel(self):
        """EDF stores record-major; a channel must come back contiguous in time."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.edf")
            sig = np.vstack(
                [np.arange(12, dtype=np.int64), -np.arange(12, dtype=np.int64)]
            )
            write_edf(path, nsig=2, nrec=3, persig=4, signals=sig)
            got = edf2jeeg(path)["EEGData"]
            np.testing.assert_array_equal(got[0], np.arange(12))
            np.testing.assert_array_equal(got[1], -np.arange(12))

    def test_calibration_maps_digital_to_physical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.edf")
            write_edf(path)
            ch = edf2jeeg(path)["EEGChannels"][0]
            # digital [-32768, 32767] -> physical [-1000, 1000]
            self.assertAlmostEqual(ch["DigitalMinimum"] * ch["Gain"] + ch["Offset"], -1000.0, 3)
            self.assertAlmostEqual(ch["DigitalMaximum"] * ch["Gain"] + ch["Offset"], 1000.0, 3)
            self.assertEqual(ch["Unit"], "uV")
            self.assertEqual(ch["Label"], "ch1")

    def test_bdf_24bit_is_sign_extended(self):
        """The bug worth a test: 24-bit samples must not read as huge positives."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.bdf")
            sig = np.array([[-1, -2, 100, 8388607], [1, 2, -100, -8388608]], dtype=np.int64)
            write_edf(path, nsig=2, nrec=1, persig=4, bdf=True, signals=sig)
            got = edf2jeeg(path)
            self.assertEqual(got["EEGHeader"]["Format"], "BDF")
            np.testing.assert_array_equal(got["EEGData"], sig)

    def test_record_count_is_taken_from_the_file_when_the_header_lies(self):
        """EDF+ allows -1 for unknown; the file size is the authority."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.edf")
            expect = write_edf(path, nsig=2, nrec=3, persig=4)

            with open(path, "r+b") as fid:  # overwrite the record count with -1
                fid.seek(236)
                fid.write(("%-8s" % "-1").encode())

            got = edf2jeeg(path)
            self.assertEqual(got["EEGHeader"]["NumberOfDataRecords"], 3)
            np.testing.assert_array_equal(got["EEGData"], expect)

    def test_a_lookalike_extension_is_rejected_clearly(self):
        """EyeLink also uses .edf; the message must say so, not fail in a field."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eyelink.edf")

            with open(path, "wb") as fid:
                fid.write(b"**CONFIG\x00" + b"\x00" * 400)

            with self.assertRaises(ValueError) as ctx:
                edf2jeeg(path)

            self.assertIn("not an EDF/BDF", str(ctx.exception))


class TestBrainVision(unittest.TestCase):
    def test_multiplexed_is_deinterleaved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.vhdr")
            expect = write_brainvision(path, nchan=3, nsamp=50)
            got = bv2jeeg(path)
            np.testing.assert_allclose(got["EEGData"], expect)
            self.assertEqual(got["EEGHeader"]["NumberOfChannels"], 3)
            self.assertEqual(got["EEGHeader"]["NumberOfSamples"], 50)
            self.assertAlmostEqual(got["EEGHeader"]["SamplingFrequency"], 2000.0)

    def test_vectorized_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.vhdr")
            expect = write_brainvision(path, nchan=3, nsamp=40, orientation="VECTORIZED")
            np.testing.assert_allclose(bv2jeeg(path)["EEGData"], expect)

    def test_channel_labels_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.vhdr")
            write_brainvision(path, nchan=2, nsamp=10)
            ch = bv2jeeg(path)["EEGChannels"]
            self.assertEqual([c["Label"] for c in ch], ["E1", "E2"])
            self.assertEqual(ch[0]["Unit"], "uV")

    def test_missing_binary_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.vhdr")
            write_brainvision(path, nchan=2, nsamp=10)
            os.remove(os.path.join(tmp, "b.eeg"))

            with self.assertRaises(ValueError):
                bv2jeeg(path)


class TestDispatch(unittest.TestCase):
    def test_extension_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            edf = os.path.join(tmp, "a.edf")
            write_edf(edf)
            self.assertEqual(eeg2jeeg(edf)["EEGHeader"]["Format"], "EDF")

            vhdr = os.path.join(tmp, "b.vhdr")
            write_brainvision(vhdr)
            self.assertEqual(eeg2jeeg(vhdr)["EEGHeader"]["Format"], "BrainVision")

    def test_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            eeg2jeeg("/nonexistent/file.xyz")


class TestJDataRoundTrip(unittest.TestCase):
    """A JEEG structure has to survive the encode/decode it exists for."""

    def test_samples_survive_a_compressed_round_trip(self):
        import jdata as jd

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.edf")
            expect = write_edf(path, nsig=3, nrec=5, persig=8)
            jeeg = edf2jeeg(path)
            encoded = jd.encode(
                {"rec": jeeg}, compression="zlib", compressarraysize=0, nthread=4
            )
            back = jd.decode(encoded, nthread=4)["rec"]
            np.testing.assert_array_equal(back["EEGData"], expect)
            self.assertEqual(
                back["EEGHeader"]["NumberOfSignals"], jeeg["EEGHeader"]["NumberOfSignals"]
            )



class TestBrainVisionWriter(unittest.TestCase):
    """BrainVision is a three-file format; all three must come back."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vhdr = os.path.join(self.tmp, "sub-01_eeg.vhdr")
        write_brainvision(self.vhdr, nchan=4, nsamp=64)
        with open(os.path.join(self.tmp, "sub-01_eeg.vmrk"), "wb") as fid:
            fid.write(
                b"Brain Vision Data Exchange Marker File\n[Common Infos]\n"
                b"DataFile=sub-01_eeg.eeg\n[Marker Infos]\nMk1=New Segment,,1,1,0\n"
            )
        raw = open(self.vhdr, "rb").read()
        if b"MarkerFile" not in raw:
            raw = raw.replace(b"DataFile=", b"MarkerFile=sub-01_eeg.vmrk\nDataFile=", 1)
            open(self.vhdr, "wb").write(raw)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _digests(self, folder):
        out = {}
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                with open(full, "rb") as fid:
                    out[name] = hashlib.sha256(fid.read()).hexdigest()
        return out

    def test_marker_file_is_captured(self):
        self.assertIn("EEGSourceMarker", bv2jeeg(self.vhdr))

    def test_roundtrip_all_three_files(self):
        before = self._digests(self.tmp)
        out = os.path.join(self.tmp, "rebuilt")
        os.makedirs(out)
        got = jeeg2bv(bv2jeeg(self.vhdr), os.path.join(out, "sub-01_eeg.vhdr"))
        self.assertEqual(set(got), set(before))
        self.assertEqual(got, before)

    def test_vectorized_layout_roundtrips(self):
        raw = open(self.vhdr, "rb").read().replace(b"MULTIPLEXED", b"VECTORIZED")
        open(self.vhdr, "wb").write(raw)
        before = self._digests(self.tmp)
        out = os.path.join(self.tmp, "rebuilt")
        os.makedirs(out)
        got = jeeg2bv(bv2jeeg(self.vhdr), os.path.join(out, "sub-01_eeg.vhdr"))
        self.assertEqual(got["sub-01_eeg.eeg"], before["sub-01_eeg.eeg"])


class TestEEGLABFdtFallback(unittest.TestCase):
    """A .set renamed into BIDS layout still names its old .fdt."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fallback_resolves_by_stem(self):
        from jdata.jeeg import _eeglab_fdt

        setpath = os.path.join(self.tmp, "sub-01_task-x_ieeg.set")
        open(setpath, "wb").write(b"stub")
        want = os.path.join(self.tmp, "sub-01_task-x_ieeg.fdt")
        open(want, "wb").write(b"data")
        self.assertEqual(_eeglab_fdt(setpath, "S_1_cond1_run1.fdt"), want)

    def test_missing_companion_still_raises(self):
        from jdata.jeeg import _eeglab_fdt

        setpath = os.path.join(self.tmp, "sub-01_task-x_ieeg.set")
        open(setpath, "wb").write(b"stub")
        self.assertRaises(ValueError, _eeglab_fdt, setpath, "nowhere.fdt")


class TestEegInfoPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_edf_summary_has_no_phi(self):
        from jdata.njencode import PHI_KEYS, SIDECAR_KEYS

        path = os.path.join(self.tmp, "x.edf")
        write_edf(path, nsig=3, nrec=4, persig=8)
        info = eeginfo(path)
        self.assertEqual(info["Format"], "EDF")
        self.assertFalse([k for k in info if k in PHI_KEYS or k in SIDECAR_KEYS])

    def test_edf_plus_discontinuous_is_reported(self):
        path = os.path.join(self.tmp, "d.edf")
        write_edf(path, nsig=2, nrec=3, persig=8)
        with open(path, "r+b") as fid:
            fid.seek(192)
            fid.write(b"EDF+D".ljust(44))
        info = eeginfo(path)
        self.assertEqual(info["Format"], "EDF+")
        self.assertEqual(info["Continuity"], "discontinuous")


class TestGzippedSources(unittest.TestCase):
    """.edf.gz is the same payload; only the source file name differs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_edf_gz_reads_as_edf(self):
        import gzip
        from jdata.njencode import encode_attachment, encoder_for

        plain = os.path.join(self.tmp, "x.edf")
        write_edf(plain, nsig=3, nrec=4, persig=8)
        gzpath = os.path.join(self.tmp, "x.edf.gz")
        with open(plain, "rb") as src, gzip.open(gzpath, "wb") as dst:
            dst.write(src.read())
        self.assertEqual(encoder_for(".edf.gz"), encoder_for(".edf"))
        _h, payload, ext, keys = encode_attachment(gzpath, ".edf.gz", compression="zlib")
        self.assertEqual(ext, ".beeg")
        self.assertEqual(keys, ("EEGData",))
        import jdata as jd

        back = jd.loadbs(payload)
        self.assertTrue(back["EEGSource"]["Gzipped"])
        self.assertEqual(back["EEGSource"]["OriginalName"], "x.edf.gz")

    def test_gz_and_plain_give_the_same_samples(self):
        import gzip
        import numpy as np
        from jdata.njencode import _load

        plain = os.path.join(self.tmp, "x.edf")
        write_edf(plain, nsig=3, nrec=4, persig=8)
        gzpath = os.path.join(self.tmp, "x.edf.gz")
        with open(plain, "rb") as src, gzip.open(gzpath, "wb") as dst:
            dst.write(src.read())
        self.assertTrue(
            np.array_equal(_load(plain, ".edf")["EEGData"], _load(gzpath, ".edf.gz")["EEGData"])
        )


if __name__ == "__main__":
    unittest.main()


class TestVariantAndRoundTrip(unittest.TestCase):
    """EDF, EDF+C, EDF+D and BDF must be told apart, and rebuilt exactly.

    Encoding does not keep the original bytes in the content store, so the
    wrapper is the only surviving description of the file. If it cannot
    reproduce the file it has silently lost data.
    """

    def _with_reserved(self, path, reserved, **kw):
        expect = write_edf(path, **kw)

        with open(path, "r+b") as fid:
            fid.seek(192)
            fid.write(("%-44s" % reserved).encode("latin-1"))

        return expect

    def test_plain_edf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.edf")
            write_edf(p)
            h = edf2jeeg(p)["EEGHeader"]
            self.assertEqual(h["Format"], "EDF")
            self.assertEqual(h["Continuity"], "continuous")

    def test_edf_plus_continuous_and_discontinuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            for reserved, expect in (("EDF+C", "continuous"), ("EDF+D", "discontinuous")):
                p = os.path.join(tmp, "%s.edf" % reserved)
                self._with_reserved(p, reserved)
                h = edf2jeeg(p)["EEGHeader"]
                self.assertEqual(h["Format"], "EDF+", reserved)
                self.assertEqual(h["Continuity"], expect, reserved)

    def test_bdf_is_not_reported_as_edf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.bdf")
            write_edf(p, bdf=True)
            self.assertEqual(edf2jeeg(p)["EEGHeader"]["Format"], "BDF")

    def test_source_block_records_the_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.edf")
            write_edf(p, nsig=2)
            src = edf2jeeg(p)["EEGSource"]
            self.assertEqual(src["Bytes"], os.path.getsize(p))
            self.assertEqual(src["File"], "a.edf")
            self.assertEqual(len(src["SHA256"]), 64)
            # the header is kept verbatim: 256 + 256 per signal
            self.assertEqual(len(src["RawHeader"]["_ByteStream_"]), 256 + 256 * 2)

    def test_round_trip_is_byte_exact(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            for name, kw in (
                ("edf", {}),
                ("bdf", {"bdf": True}),
                ("wide", {"nsig": 5, "nrec": 7, "persig": 3}),
            ):
                p = os.path.join(tmp, name + ".edf")
                write_edf(p, **kw)
                jeeg = edf2jeeg(p)
                out = os.path.join(tmp, name + ".rebuilt")
                digest = jeeg2edf(jeeg, out)
                with open(p, "rb") as fid:
                    original = hashlib.sha256(fid.read()).hexdigest()
                self.assertEqual(digest, original, name)
                self.assertEqual(digest, jeeg["EEGSource"]["SHA256"], name)
                with open(p, "rb") as a, open(out, "rb") as b:
                    self.assertEqual(a.read(), b.read(), name)

    def test_rebuild_refuses_without_the_raw_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.edf")
            write_edf(p)
            jeeg = edf2jeeg(p)
            jeeg["EEGSource"].pop("RawHeader")

            with self.assertRaises(ValueError):
                jeeg2edf(jeeg, os.path.join(tmp, "out.edf"))

    def test_channels_keep_their_original_index(self):
        """Needed to put annotation channels back where they were."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.edf")
            write_edf(p, nsig=3)
            ch = edf2jeeg(p)["EEGChannels"]
            self.assertEqual([c["OriginalIndex"] for c in ch], [0, 1, 2])

