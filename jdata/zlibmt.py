"""
Multi-threaded zlib compression producing a standard, single zlib stream.

zlib is the right default for a data-sharing format: it is native to MATLAB,
Python, R, Julia, every browser and every language runtime, with nothing to
install.  Its one drawback against a modern codec is speed, and that is an
implementation limit rather than a format limit -- so this module removes it.

The method is the one ``pigz`` uses.  Input is cut into fixed-size blocks and
each block is deflated independently, terminated with ``Z_FULL_FLUSH``, which
resets the compressor's sliding window so the block carries no dependency on the
one before it.  The blocks are then concatenated, a final empty terminating
block is appended, and the standard 2-byte zlib header and Adler-32 trailer are
wrapped around the result.  What comes out is an ordinary zlib stream: any
``zlib.decompress`` -- or ``inflate`` in any other language -- reads it without
knowing it was produced in parallel.

Two properties matter here:

**Determinism.**  Block boundaries depend only on ``blocksize``, never on the
thread count, so the output bytes are a pure function of (data, level,
blocksize).  The same input compresses to the same bytes on a 4-core machine and
a 128-core one, which is what lets a content fingerprint stay reproducible.

**Real parallelism.**  ``zlib.compress`` releases the GIL, so threads genuinely
run concurrently and no subprocess or shared memory is involved.

The cost is a slightly larger output: resetting the window at every boundary
loses cross-block matches.  With the default 4 MiB block that is a fraction of a
percent on volumetric data.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>
"""

import os
import zlib

__all__ = [
    "compress",
    "decompress",
    "decompress_parallel",
    "gzip_compress",
    "DEFAULT_BLOCKSIZE",
]

#: 4 MiB: large enough that the lost cross-block matches are negligible, small
#: enough that a typical volume still splits across the available threads
DEFAULT_BLOCKSIZE = 4 << 20

_FLEVEL = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 3, 9: 3}


def _zlib_header(level):
    """The 2-byte zlib header for a 32 KiB window at this level."""
    cmf = 0x78  # CM=8 (deflate), CINFO=7 (32 KiB window)
    flevel = _FLEVEL.get(level if level >= 0 else 6, 2)
    flg = flevel << 6
    # FCHECK is chosen so that (CMF << 8 | FLG) is a multiple of 31
    flg |= 31 - ((cmf << 8 | flg) % 31)
    return bytes((cmf, flg))


def _deflate_block(args):
    """Deflate one block into a self-contained, FULL_FLUSH-terminated segment."""
    chunk, level = args
    engine = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return engine.compress(chunk) + engine.flush(zlib.Z_FULL_FLUSH)


def compress(
    data, level=6, nthread=None, blocksize=DEFAULT_BLOCKSIZE, return_offsets=False
):
    """Compress ``data`` to a standard zlib stream, using threads.

    Parameters
    ----------
    data : bytes-like
    level : int
        zlib compression level, 0-9.
    nthread : int, optional
        Worker threads; defaults to the CPU count, capped by the block count.
        Affects only how fast the output is produced, never what it is: ``1``
        takes the same block-wise path as ``32``.
    blocksize : int
        Bytes per independently-deflated block.  Affects the output bytes, so it
        is part of the reproducibility contract and should not be varied between
        runs that must agree.

    return_offsets : bool
        Also return the block index: ``[[compressed, uncompressed], ...]``, one
        pair per block plus a final sentinel carrying the totals.  DEFLATE
        records nothing about where its blocks begin, so this index is the only
        way a reader can inflate them concurrently -- see
        :func:`decompress_parallel`.

    Returns
    -------
    bytes, or ``(bytes, list)`` when ``return_offsets``
        A valid zlib stream, decompressible by any standard implementation.
        Not byte-identical to ``zlib.compress`` output -- the block structure
        differs -- but decompressing to exactly the same data.
    """
    view = memoryview(bytes(data)) if not isinstance(data, (bytes, bytearray)) else memoryview(data)
    total = len(view)
    if nthread is None:
        nthread = os.cpu_count() or 1
    nthread = max(1, int(nthread))

    # The block-wise construction is used unconditionally, including for a
    # single thread.  Falling back to zlib.compress when nthread == 1 would make
    # the output bytes depend on how many threads the caller happened to use,
    # and the whole point is that they depend only on (data, level, blocksize):
    # an attachment must hash the same whoever produced it.
    blocks = [view[i : i + blocksize] for i in range(0, total, blocksize)] or [b""]
    workers = max(1, min(nthread, len(blocks)))
    payloads = [(bytes(b), level) for b in blocks]

    if workers == 1:
        segments = [_deflate_block(item) for item in payloads]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            segments = list(pool.map(_deflate_block, payloads))

    tail = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    terminator = tail.compress(b"") + tail.flush(zlib.Z_FINISH)

    out = bytearray(_zlib_header(level))
    offsets = []
    uncompressed = 0
    for index, segment in enumerate(segments):
        offsets.append([len(out), uncompressed])
        out += segment
        uncompressed += len(blocks[index])
    offsets.append([len(out), uncompressed])  # sentinel: end of the last block
    out += terminator
    out += (zlib.adler32(view) & 0xFFFFFFFF).to_bytes(4, "big")
    return (bytes(out), offsets) if return_offsets else bytes(out)


def decompress(data):
    """Decompress a zlib stream.  Plain :func:`zlib.decompress`; here for symmetry."""
    return zlib.decompress(data)


def decompress_parallel(data, offsets, nthread=None):
    """Inflate a block-indexed stream concurrently.

    This is the half DEFLATE cannot do unaided.  Compression parallelises by
    splitting the input, but decompression of an arbitrary zlib stream cannot:
    the format is a bit-oriented stream with no framing, so block *N* is
    unreachable without inflating 1..N-1, and LZ77 back-references reach 32 KiB
    into previously *decompressed* output, so even a known block start needs the
    preceding window as its dictionary.

    ``Z_FULL_FLUSH`` removes the second obstacle at write time -- it resets the
    window, making blocks self-contained -- but nothing in the stream then
    records *where* the boundaries are, and a decoder cannot reliably find them:
    the sync marker it would scan for occurs by chance inside compressed data,
    and an ordinary ``zlib.compress`` stream has no reset points at all.
    ``offsets`` supplies exactly that missing information, which is why it has
    to travel beside the data rather than inside it.

    Parameters
    ----------
    data : bytes-like
        A stream from :func:`compress` or :func:`gzip_compress`.
    offsets : sequence
        ``[[compressed, uncompressed], ...]`` from those functions with
        ``return_offsets=True``; the last pair is a sentinel giving the totals.
    nthread : int, optional
        Worker threads; defaults to the CPU count, capped by the block count.

    Returns
    -------
    bytes
        Identical to a serial inflate of the same stream.

    Raises
    ------
    ValueError
        If the index does not describe the stream.  A caller should treat that
        as "fall back to a serial inflate", not as data loss: the stream is
        still an ordinary zlib stream regardless.
    """
    if offsets is None or len(offsets) < 2:
        raise ValueError("block index needs at least one block and a sentinel")
    view = (
        memoryview(data)
        if isinstance(data, (bytes, bytearray))
        else memoryview(bytes(data))
    )
    total = int(offsets[-1][1])
    if total < 0 or int(offsets[-1][0]) > len(view):
        raise ValueError("block index does not fit the stream")

    if nthread is None:
        nthread = os.cpu_count() or 1
    count = len(offsets) - 1
    workers = max(1, min(int(nthread), count))

    out = bytearray(total)
    target = memoryview(out)

    def work(index):
        cstart, ustart = int(offsets[index][0]), int(offsets[index][1])
        cend, uend = int(offsets[index + 1][0]), int(offsets[index + 1][1])
        engine = zlib.decompressobj(-zlib.MAX_WBITS)
        chunk = engine.decompress(view[cstart:cend]) + engine.flush()
        if len(chunk) != uend - ustart:
            raise ValueError(
                "block %d inflated to %d bytes, index declares %d"
                % (index, len(chunk), uend - ustart)
            )
        # disjoint slices of a fixed-length buffer, so this is safe to do from
        # several threads at once
        target[ustart:uend] = chunk

    if workers == 1:
        for index in range(count):
            work(index)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(work, range(count)):
                pass
    return bytes(out)


def gzip_compress(
    data,
    level=6,
    nthread=None,
    blocksize=DEFAULT_BLOCKSIZE,
    mtime=0,
    return_offsets=False,
):
    """Same method, wrapped in a gzip container instead of a zlib one.

    ``mtime`` defaults to zero rather than the current time, so the output is
    reproducible -- a timestamp in the header would change the bytes on every
    run and defeat content addressing.
    """
    view = memoryview(bytes(data)) if not isinstance(data, (bytes, bytearray)) else memoryview(data)
    total = len(view)
    if nthread is None:
        nthread = os.cpu_count() or 1
    nthread = max(1, int(nthread))

    header = bytes((0x1F, 0x8B, 8, 0)) + int(mtime).to_bytes(4, "little") + bytes((0, 0xFF))
    trailer = (zlib.crc32(bytes(view)) & 0xFFFFFFFF).to_bytes(4, "little") + (
        total & 0xFFFFFFFF
    ).to_bytes(4, "little")

    blocks = [view[i : i + blocksize] for i in range(0, total, blocksize)] or [b""]
    workers = max(1, min(nthread, len(blocks)))
    payloads = [(bytes(b), level) for b in blocks]

    if workers == 1:
        segments = [_deflate_block(item) for item in payloads]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            segments = list(pool.map(_deflate_block, payloads))

    tail = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    terminator = tail.compress(b"") + tail.flush(zlib.Z_FINISH)

    out = bytearray(header)
    offsets = []
    uncompressed = 0
    for index, segment in enumerate(segments):
        offsets.append([len(out), uncompressed])
        out += segment
        uncompressed += len(blocks[index])
    offsets.append([len(out), uncompressed])  # sentinel: end of the last block
    out += terminator
    out += trailer
    return (bytes(out), offsets) if return_offsets else bytes(out)
