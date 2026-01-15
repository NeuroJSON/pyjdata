"""Command line utility for pyjdata.

Provides a routine for converting text-/binary-based JData files to Python data.

Copyright (c) 2019-2026 Qianqian Fang <q.fang at neu.edu>

Call

    python -mjdata -h

to get help with command line usage.
"""

import argparse
import os
from sys import exit  # pylint: disable=redefined-builtin

from .jfile import loadjd, savejd, jext


def main():
    #
    # get arguments and invoke the conversion routines
    #

    parser = argparse.ArgumentParser(
        description="Convert a text JSON/JData file to a binary JSON/JData file and vice versa."
    )

    parser.add_argument(
        "file",
        nargs="+",
        help="path to a text-JData (json/jdt/jnii/jmsh/jnirs) file or a binary JData (bjd/jdb/bnii/bmsh/bnirs) file",
    )

    parser.add_argument(
        "-t",
        "--indent",
        type=int,
        help="JSON indentation size",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_const",
        const=True,
        default=False,
        help="overwrite existing files when converting",
    )

    parser.add_argument(
        "-c",
        "--compression",
        default="zlib",
        help="set compression method (zlib, gzip, lzma, lz4)",
    )

    parser.add_argument(
        "-O",
        "--outdir",
        help="output directory",
    )

    parser.add_argument(
        "-s",
        "--suffix",
        help="output file suffix",
    )

    args = parser.parse_args()

    for path in args.file:
        pathname, fullfilename = os.path.split(path)
        filename, extname = os.path.splitext(fullfilename)
        ext = extname.lower()

        if pathname is None:
            pathname = "."

        if ext in jext["t"]:
            dest = os.path.join(
                (args.outdir if args.outdir else pathname),
                filename + (args.suffix if args.suffix else ".jdb"),
            )
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception("File {} already exists.".format(dest))
                print(f"converting '{path}' to '{dest}'")

                data = loadjd(path)
                savejd(data, dest, **(vars(args)))

            except Exception as e:
                print("Error: {}".format(e))
                exit(1)

        elif ext in jext["b"]:
            dest = os.path.join(
                (args.outdir if args.outdir else pathname),
                filename + (args.suffix if args.suffix else ".json"),
            )
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception("File {} already exists.".format(dest))
                data = loadjd(path)
                savejd(data, dest, **(vars(args)))

            except RuntimeError as e:
                print("Error: {}".format(e))
                exit(1)
        else:
            print("Unsupported file extension on file: {}".format(path))
            exit(1)


if __name__ == "__main__":
    exit(main())
