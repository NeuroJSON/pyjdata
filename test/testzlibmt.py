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

from jdata.zlibmt import DEFAULT_BLOCKSIZE, compress, decompress, gzip_compress


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


if __name__ == "__main__":
    unittest.main()
