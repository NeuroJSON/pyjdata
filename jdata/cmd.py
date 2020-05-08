"""Command line utility for pyjdata.

Provides a routine for converting text-/binary-based JData files to Python data.

Call

    python -m jdata.cmd -h

to get help with command line usage.
"""

import argparse
import os
import sys

from jdata import load, save


def main():
    #
    # get arguments and invoke the conversion routines
    #

    parser = argparse.ArgumentParser(
        description='Convert a text JData file to a binary JData file and vice versa.')

    parser.add_argument(
        'file', nargs='+',
        help='path to a text-JData (.json, .jdat) file or a binary JData (.bjd, .jbat) file')
    parser.add_argument(
        '-f', '--force', action='store_const', const=True,
        default=False, help='overwrite existing files when converting')
    args = parser.parse_args()

    for path in args.file:
        spl = os.path.splitext(path)
        ext = spl[1].lower()

        if ext == '.json' or ext == '.jdat':
            dest = spl[0] + '.jbat'
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception('File {} already exists.'.format(dest))
                data = load(path)
                save(data, dest)
                if args.remove_input:
                    os.remove(path)
            except Exception as e:
                print('Error: {}'.format(e))
                sys.exit(1)

        elif ext == '.bjd' or ext == '.jbat':
            dest = spl[0] + '.json'
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception('File {} already exists.'.format(dest))
                data = load(path)
                save(data,dest)
                if args.remove_input:
                    os.remove(path)
            except RuntimeError as e:
                print('Error: {}'.format(e))
                sys.exit(1)
        else:
            print('Unsupported file extension on file: {}'.format(path))
            sys.exit(1)

if __name__ == '__main__':
    main()
