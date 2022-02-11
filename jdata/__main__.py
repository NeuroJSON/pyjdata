"""Command line utility for pyjdata.

Provides a routine for converting text-/binary-based JData files to Python data.

Call

    python -mjdata -h

to get help with command line usage.
"""

import argparse
import os
import sys

from . import load, save, jext


def main():
    #
    # get arguments and invoke the conversion routines
    #

    parser = argparse.ArgumentParser(
        description='Convert a text JData file to a binary JData file and vice versa.')

    parser.add_argument(
        'file', nargs='+',
        help='path to a text-JData (.json, .jdt) file or a binary JData (.bjd, .jdb) file')
    parser.add_argument(
        '-f', '--force', action='store_const', const=True,
        default=False, help='overwrite existing files when converting')
    parser.add_argument(
        '-r', '--remove_input', action='store_const', const=True,
        default=False, help='delete the input file name after conversion')
    parser.add_argument(
        '-c', '--compression', default='', help='compress large records')
    parser.add_argument(
        '-n', '--compressarraysize', default=100, help='compress large records')

    args = parser.parse_args()

    for path in args.file:
        spl = os.path.splitext(path)
        ext = spl[1].lower()

        if(ext in jext['t']):
            dest = spl[0] + '.jdb'
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception('File {} already exists.'.format(dest))
                data = load(path)
                if(len(args.compression)>0):
                    save(data,dest,{'compression':args.compression})
                else:
                    save(data,dest)
                if args.remove_input:
                    os.remove(path)
            except Exception as e:
                print('Error: {}'.format(e))
                sys.exit(1)

        elif(ext in jext['b']):
            dest = spl[0] + '.json'
            try:
                if os.path.exists(dest) and not args.force:
                    raise Exception('File {} already exists.'.format(dest))
                data = load(path)
                if(len(args.compression)>0):
                    save(data,dest,{'compression':args.compression})
                else:
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
