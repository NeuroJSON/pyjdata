"""Tests for jdata.zlibmt -- multi-threaded zlib and gzip.

zlib is the portable choice for a data-sharing format: native to MATLAB,
Python, R, Julia and every browser, with nothing to install.  Its only real
drawback against a modern codec is speed, and that is an implementation limit,
not a format limit.  The three properties that make removing it safe:

  * the output is an **ordinary** stream, readable by any decompressor that has
    never heard of this module;
  * the output is **deterministic** -- a pure function of (data, level,
    blocksize), independent of the thread count -- so a content fingerprint
    computed over it stays reproducible;
  * the size penalty from resetting the window at block boundaries is tiny.
"""

import os
import gzip
import sys
import zlib
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import jdata as jd
from jdata.zlibmt import (
    DEFAULT_BLOCKSIZE,
    compress,
    decompress,
    decompress_parallel,
    gzip_compress,
)


def _payload(nbytes, seed=0):
    """Compressible but not trivial: looks like volumetric integer data."""
    count = nbytes // 2
    return np.random.RandomState(seed).randint(0, 4096, size=count, dtype=np.uint16).tobytes()


class TestZlibStandardCompliance(unittest.TestCase):
    def test_stdlib_zlib_reads_the_output(self):
        data = _payload(12 << 20)
        self.assertEqual(zlib.decompress(compress(data, nthread=8)), data)

    def test_incremental_decompressobj_reads_the_output(self):
        """A streaming reader must cope too, not just the one-shot helper."""
        data = _payload(12 << 20)
        engine = zlib.decompressobj()
        blob = compress(data, nthread=8)
        out = b"".join(engine.decompress(blob[i : i + 7919]) for i in range(0, len(blob), 7919))
        self.assertEqual(out + engine.flush(), data)

    def test_header_is_a_valid_zlib_header(self):
        for level in (1, 3, 6, 9):
            blob = compress(_payload(9 << 20), level=level, nthread=4)
            self.assertEqual(blob[0], 0x78)
            self.assertEqual((blob[0] << 8 | blob[1]) % 31, 0)

    def test_adler32_trailer_is_correct(self):
        data = _payload(9 << 20)
        blob = compress(data, nthread=4)
        self.assertEqual(int.from_bytes(blob[-4:], "big"), zlib.adler32(data) & 0xFFFFFFFF)

    def test_our_decompress_matches_stdlib(self):
        data = _payload(9 << 20)
        self.assertEqual(decompress(compress(data, nthread=4)), data)


class TestDeterminism(unittest.TestCase):
    def test_output_does_not_depend_on_thread_count(self):
        data = _payload(20 << 20)
        reference = compress(data, nthread=2)
        for threads in (1, 3, 8, 16, 64):
            self.assertEqual(
                compress(data, nthread=threads),
                reference,
                "thread count %d changed the output" % threads,
            )

    def test_repeated_calls_are_identical(self):
        data = _payload(9 << 20)
        self.assertEqual(compress(data, nthread=8), compress(data, nthread=8))

    def test_level_changes_the_output(self):
        data = _payload(9 << 20)
        self.assertNotEqual(compress(data, level=1, nthread=8), compress(data, level=9, nthread=8))

    def test_blocksize_is_part_of_the_contract(self):
        """Documented as affecting the bytes, so it must not be varied silently."""
        data = _payload(20 << 20)
        self.assertNotEqual(
            compress(data, nthread=8, blocksize=4 << 20),
            compress(data, nthread=8, blocksize=8 << 20),
        )


class TestEdgeCases(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(zlib.decompress(compress(b"", nthread=8)), b"")

    def test_input_smaller_than_one_block(self):
        data = b"short payload"
        self.assertEqual(zlib.decompress(compress(data, nthread=8)), data)

    def test_input_exactly_one_block(self):
        data = _payload(DEFAULT_BLOCKSIZE)
        self.assertEqual(zlib.decompress(compress(data, nthread=8)), data)

    def test_input_one_byte_over_a_block(self):
        data = _payload(DEFAULT_BLOCKSIZE) + b"x"
        self.assertEqual(zlib.decompress(compress(data, nthread=8)), data)

    def test_single_thread_uses_the_same_construction(self):
        """One thread must not be a special case, or the bytes stop being stable."""
        data = _payload(9 << 20)
        self.assertEqual(compress(data, nthread=1), compress(data, nthread=8))
        self.assertEqual(zlib.decompress(compress(data, nthread=1)), data)

    def test_accepts_memoryview_and_bytearray(self):
        data = _payload(9 << 20)
        for form in (bytearray(data), memoryview(data)):
            self.assertEqual(zlib.decompress(compress(form, nthread=4)), data)

    def test_incompressible_input(self):
        data = os.urandom(9 << 20)
        self.assertEqual(zlib.decompress(compress(data, nthread=8)), data)

    def test_highly_compressible_input(self):
        data = b"\x00" * (20 << 20)
        blob = compress(data, nthread=8)
        self.assertEqual(zlib.decompress(blob), data)
        self.assertLess(len(blob), len(data) // 100)


class TestSizePenalty(unittest.TestCase):
    def test_block_reset_costs_almost_nothing(self):
        data = _payload(40 << 20)
        serial = zlib.compress(data, 6)
        parallel = compress(data, level=6, nthread=16)
        overhead = (len(parallel) - len(serial)) / len(serial)
        self.assertLess(overhead, 0.01, "block boundaries cost %.3f%%" % (overhead * 100))


class TestGzipContainer(unittest.TestCase):
    def test_stdlib_gzip_reads_the_output(self):
        data = _payload(12 << 20)
        self.assertEqual(gzip.decompress(gzip_compress(data, nthread=8)), data)

    def test_deterministic_across_thread_counts(self):
        data = _payload(20 << 20)
        self.assertEqual(gzip_compress(data, nthread=3), gzip_compress(data, nthread=32))

    def test_no_timestamp_in_the_header(self):
        """A timestamp would change the bytes every run and break addressing."""
        data = _payload(9 << 20)
        self.assertEqual(gzip_compress(data, nthread=4)[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(gzip_compress(data, nthread=4), gzip_compress(data, nthread=4))

    def test_trailer_carries_crc32_and_length(self):
        data = _payload(9 << 20)
        blob = gzip_compress(data, nthread=4)
        self.assertEqual(int.from_bytes(blob[-8:-4], "little"), zlib.crc32(data) & 0xFFFFFFFF)
        self.assertEqual(int.from_bytes(blob[-4:], "little"), len(data) & 0xFFFFFFFF)

    def test_small_and_empty_inputs(self):
        for data in (b"", b"tiny"):
            self.assertEqual(gzip.decompress(gzip_compress(data, nthread=8)), data)


class TestJdataIntegration(unittest.TestCase):
    """nthread must reach the codec through jdata.encode.

    Regression test: the dense-ndarray path had its own inline codec dispatch
    that called zlib.compress directly, so the common case silently ignored
    nthread however it was passed.
    """

    def setUp(self):
        import jdata

        self.jd = jdata
        self.array = np.random.RandomState(3).randint(0, 4096, size=6_000_000, dtype=np.uint16)

    def test_encoded_payload_is_identical_however_many_threads(self):
        """Any explicit nthread selects the block-wise path, so bytes agree."""
        one = self.jd.encode({"v": self.array}, compression="zlib", compressarraysize=0, nthread=1)
        many = self.jd.encode(
            {"v": self.array}, compression="zlib", compressarraysize=0, nthread=16
        )
        self.assertEqual(one["v"]["_ArrayZipData_"], many["v"]["_ArrayZipData_"])

    def test_omitting_nthread_keeps_the_historical_output(self):
        """Existing callers must not see their produced bytes change."""
        import zlib as _zlib

        legacy = self.jd.encode({"v": self.array}, compression="zlib", compressarraysize=0)
        self.assertEqual(legacy["v"]["_ArrayZipData_"], _zlib.compress(self.array.tobytes()))

    def test_round_trip_through_the_threaded_path(self):
        encoded = self.jd.encode(
            {"v": self.array}, compression="zlib", compressarraysize=0, nthread=16
        )
        self.assertTrue(np.array_equal(self.jd.decode(encoded)["v"], self.array))

    def test_other_codecs_still_work_after_the_dedup(self):
        for codec in ("zlib", "gzip", "lzma"):
            encoded = self.jd.encode(
                {"v": self.array[:1000]}, compression=codec, compressarraysize=0
            )
            self.assertTrue(np.array_equal(self.jd.decode(encoded)["v"], self.array[:1000]), codec)


class TestBlockIndex(unittest.TestCase):
    """``_ArrayZipOffsets_``: the block index that makes parallel inflate possible.

    DEFLATE records nothing about where its blocks begin, which is why a
    threaded decoder cannot exist for an arbitrary stream.  Writing the offsets
    down at compression time -- when they are known for free -- removes that
    obstacle without changing a byte of the stream itself.
    """

    def setUp(self):
        self.data = _payload(24 << 20, seed=7)
        self.blob, self.offsets = compress(self.data, return_offsets=True)

    def test_index_shape_and_sentinel(self):
        # [[compressed, uncompressed], ..., sentinel]: the sentinel closes the
        # last block, so every row i spans offsets[i] to offsets[i + 1]
        self.assertTrue(all(len(row) == 2 for row in self.offsets))
        # the first block starts after the 2-byte zlib header
        self.assertEqual(self.offsets[0], [2, 0])
        # the sentinel closes the last data block; the stream then carries
        # deflate's terminal empty block and the adler32 trailer
        self.assertLess(self.offsets[-1][0], len(self.blob))
        self.assertGreaterEqual(self.offsets[-1][0], len(self.blob) - 8)
        self.assertEqual(self.offsets[-1][1], len(self.data))
        for a, b in zip(self.offsets, self.offsets[1:]):
            self.assertLess(a[0], b[0])
            self.assertLess(a[1], b[1])

    def test_index_is_small(self):
        # it has to be cheap enough to sit inside the JSON document
        import json

        self.assertLess(len(json.dumps(self.offsets)), 4096)

    def test_indexing_does_not_change_the_stream(self):
        self.assertEqual(compress(self.data), self.blob)

    def test_stream_is_still_ordinary_zlib(self):
        self.assertEqual(zlib.decompress(self.blob), self.data)

    def test_parallel_inflate_matches_serial(self):
        for nthread in (1, 2, 8, 32):
            self.assertEqual(
                decompress_parallel(self.blob, self.offsets, nthread=nthread),
                self.data,
                nthread,
            )

    def test_index_locates_one_block_independently(self):
        # the point of the index: block i can be inflated without inflating 0..i-1
        i = len(self.offsets) // 2
        (cstart, ustart), (cend, uend) = self.offsets[i], self.offsets[i + 1]
        block = zlib.decompressobj(-zlib.MAX_WBITS).decompress(self.blob[cstart:cend])
        self.assertEqual(block, self.data[ustart:uend])

    def test_rejects_an_index_that_does_not_match(self):
        # a wrong index must fail loudly rather than return wrong bytes; the
        # caller falls back to a serial inflate, which always works
        for bad in (
            [[0, 0], [7, len(self.data)], [len(self.blob), len(self.data)]],
            [[0, 0], [len(self.blob), len(self.data) + 1]],
            [list(reversed(row)) for row in self.offsets],
        ):
            with self.assertRaises((ValueError, zlib.error)):
                decompress_parallel(self.blob, bad, nthread=4)

    def test_single_block_input_is_not_indexed(self):
        small = _payload(4096, seed=8)
        blob, offsets = compress(small, return_offsets=True)
        self.assertEqual(zlib.decompress(blob), small)
        self.assertEqual(decompress_parallel(blob, offsets, nthread=4), small)

    def test_gzip_container_carries_an_index_too(self):
        blob, offsets = gzip_compress(self.data, return_offsets=True)
        self.assertEqual(gzip.decompress(blob), self.data)
        self.assertEqual(decompress_parallel(blob, offsets, nthread=8), self.data)


class TestZipOffsetsInJdata(unittest.TestCase):
    """The index round-tripping through ``jd.encode``/``jd.decode``."""

    @classmethod
    def setUpClass(cls):
        import jdata as jd

        cls.jd = jd
        cls.array = np.random.RandomState(11).randint(
            0, 4096, size=12 << 20, dtype=np.uint16
        )

    def _encode(self, **kwargs):
        opt = dict(compression="zlib", compressarraysize=0)
        opt.update(kwargs)
        return self.jd.encode({"v": self.array}, **opt)["v"]

    def test_threaded_encode_emits_the_key(self):
        rec = self._encode(nthread=16)
        self.assertIn("_ArrayZipOffsets_", rec)
        self.assertGreater(len(rec["_ArrayZipOffsets_"]), 2)

    def test_untouched_when_nthread_is_omitted(self):
        # no threading asked for, no annotation: old readers see the old document
        self.assertNotIn("_ArrayZipOffsets_", self._encode())

    def test_key_does_not_depend_on_thread_count(self):
        self.assertEqual(
            self._encode(nthread=4)["_ArrayZipOffsets_"],
            self._encode(nthread=32)["_ArrayZipOffsets_"],
        )

    def test_decode_is_identical_serial_or_parallel(self):
        encoded = {"v": self._encode(nthread=16)}
        for nthread in (1, 4, 32):
            self.assertTrue(
                np.array_equal(self.jd.decode(encoded, nthread=nthread)["v"], self.array),
                nthread,
            )

    def test_an_unaware_reader_still_decodes(self):
        # strip the annotation, as a reader that does not know the key would
        rec = self._encode(nthread=16)
        rec.pop("_ArrayZipOffsets_")
        self.assertTrue(np.array_equal(self.jd.decode({"v": rec})["v"], self.array))

    def test_a_corrupt_index_falls_back_instead_of_failing(self):
        rec = self._encode(nthread=16)
        rec["_ArrayZipOffsets_"] = [[0, 0], [3, 5], [7, 11]]
        self.assertTrue(
            np.array_equal(self.jd.decode({"v": rec}, nthread=8)["v"], self.array)
        )

    def test_survives_a_json_file_round_trip(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "a.jdt")
            self.jd.save(
                {"v": self.array},
                fname,
                compression="zlib",
                compressarraysize=0,
                nthread=8,
            )
            with open(fname) as fid:
                self.assertIn("_ArrayZipOffsets_", fid.read())
            back = self.jd.load(fname, nthread=8)
        self.assertTrue(np.array_equal(back["v"], self.array))

    def test_sparse_arrays_are_indexed_too(self):
        # the decoder honours the key in both the sparse and dense branches, so
        # the encoder has to emit it in both
        import scipy.sparse

        m = scipy.sparse.random(2000, 2000, density=0.3, format="csc", random_state=1)
        rec = self.jd.encode(
            {"m": m}, compression="zlib", compressarraysize=0, nthread=8
        )["m"]
        self.assertIn("_ArrayZipOffsets_", rec)
        for nthread in (1, 16):
            back = self.jd.decode({"m": rec}, nthread=nthread)["m"]
            self.assertEqual(abs(back - m).max(), 0, nthread)

    def test_a_raw_byte_buffer_is_indexed_as_a_uint8_array(self):
        # general buffers reach the same path once viewed as a uint8 array,
        # which is how JData represents an opaque blob anyway
        buf = os.urandom(12 << 20)
        rec = self.jd.encode(
            {"b": np.frombuffer(buf, dtype=np.uint8)},
            compression="zlib",
            compressarraysize=0,
            nthread=8,
        )["b"]
        self.assertIn("_ArrayZipOffsets_", rec)
        back = self.jd.decode({"b": rec}, nthread=8)["b"]
        self.assertEqual(back.tobytes(), buf)

    def test_survives_a_bjdata_round_trip(self):
        # binary JData: the index travels as a normal integer array
        encoded = self.jd.encode(
            {"v": self.array}, compression="zlib", compressarraysize=0, nthread=8
        )
        blob = self.jd.dumpb(encoded)
        back = self.jd.loadbs(blob, nthread=8)
        self.assertTrue(np.array_equal(back["v"], self.array))



class TestCompressLevel(unittest.TestCase):
    """opt['compresslevel'] reaches zlib. The default must not move, because
    the attachment bytes are what makes a converted dataset reproducible."""

    def setUp(self):
        rng = np.random.default_rng(7)
        base = rng.integers(-2000, 2000, size=64000, dtype=np.int32)
        self.obj = {"x": np.repeat(base, 5).reshape(5, -1)}

    def _enc(self, **kw):
        return jd.encode(self.obj, compression="zlib", compressarraysize=0, **kw)["x"][
            "_ArrayZipData_"
        ]

    def test_default_is_level_six(self):
        self.assertEqual(bytes(self._enc()), bytes(self._enc(compresslevel=6)))

    def test_omitting_level_matches_historical_output(self):
        self.assertEqual(bytes(self._enc(nthread=4)), bytes(self._enc(nthread=4, compresslevel=6)))

    def test_level_changes_the_bytes(self):
        self.assertNotEqual(bytes(self._enc(compresslevel=1)), bytes(self._enc(compresslevel=9)))

    def test_every_level_round_trips(self):
        for lvl in range(10):
            enc = jd.encode(
                self.obj, compression="zlib", compressarraysize=0, compresslevel=lvl
            )
            self.assertTrue(
                np.array_equal(jd.decode(enc)["x"], self.obj["x"]), "level %d" % lvl
            )

    def test_level_applies_to_the_threaded_path_too(self):
        one = self._enc(nthread=4, compresslevel=1)
        six = self._enc(nthread=4, compresslevel=6)
        self.assertNotEqual(bytes(one), bytes(six))
        for blob in (one, six):
            enc = jd.encode(self.obj, compression="zlib", compressarraysize=0, nthread=4)
            self.assertTrue(len(blob) > 0)

    def test_threaded_output_is_independent_of_thread_count(self):
        a = self._enc(nthread=2, compresslevel=1)
        b = self._enc(nthread=8, compresslevel=1)
        self.assertEqual(bytes(a), bytes(b))



class TestDecodeThreadDefault(unittest.TestCase):
    """Reading may parallelise by default; writing may not.

    Inflating an indexed stream concurrently is bit-identical to inflating it
    serially, so there is nothing to preserve. Deflating block-wise, by
    contrast, produces different bytes from a single stream, which is why
    nthread stays opt-in on the encode side.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        base = rng.integers(-500, 500, size=1 << 21, dtype=np.int32)
        self.obj = {"x": np.repeat(base, 3)}

    def test_default_is_at_least_one(self):
        from jdata.jdata import DEFAULT_DECODE_THREADS

        self.assertGreaterEqual(DEFAULT_DECODE_THREADS, 1)

    def test_decode_default_matches_serial(self):
        enc = jd.encode(self.obj, compression="zlib", compressarraysize=0, nthread=8)
        self.assertIn("_ArrayZipOffsets_", enc["x"])
        auto = jd.decode(enc)["x"]
        serial = jd.decode(enc, nthread=1)["x"]
        self.assertTrue(np.array_equal(auto, serial))
        self.assertTrue(np.array_equal(auto, self.obj["x"]))

    def test_decode_thread_count_never_changes_output(self):
        enc = jd.encode(self.obj, compression="zlib", compressarraysize=0, nthread=8)
        want = self.obj["x"]
        for nt in (1, 2, 4, 16):
            self.assertTrue(np.array_equal(jd.decode(enc, nthread=nt)["x"], want), nt)

    def test_encode_default_stays_single_stream(self):
        """A changed encode default would move every stored byte."""
        single = jd.encode(self.obj, compression="zlib", compressarraysize=0)["x"][
            "_ArrayZipData_"
        ]
        blocked = jd.encode(self.obj, compression="zlib", compressarraysize=0, nthread=1)[
            "x"
        ]["_ArrayZipData_"]
        self.assertNotEqual(bytes(single), bytes(blocked))
        self.assertNotIn(
            "_ArrayZipOffsets_",
            jd.encode(self.obj, compression="zlib", compressarraysize=0)["x"],
        )

    def test_unindexed_payload_still_decodes(self):
        enc = jd.encode(self.obj, compression="zlib", compressarraysize=0)
        self.assertTrue(np.array_equal(jd.decode(enc)["x"], self.obj["x"]))


if __name__ == "__main__":
    unittest.main()
