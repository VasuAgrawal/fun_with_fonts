#!/usr/bin/env python3

import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--input_dir", type=pathlib.Path, required=True,
        help="Directory to crawl for files")
parser.add_argument("-o", "--output_dir", type=pathlib.Path,
        default="by_extension")
#  parser.add_argument("-e", "--extensions", nargs="+", type=str, 
#          default=[".otf", ".ttf", ".svg", ".eot", ".woff", ".woff2", ".ttc"],
        #  help="Extensions to copy")
parser.add_argument("-c", "--count", type=int, default=0,
        help="Stop after copying this many files (0 means don't stop)",
        )
args = parser.parse_args()

import os
import collections
import shutil
import json

def main():

    extensions = set()
    extension_folders = {}
    extension_mapping = collections.defaultdict(dict)
    #  for e in extensions:
    #      extension_folders[e] = args.output_dir / e[1:]
    #      extension_folders[e].mkdir(parents=True, exist_ok=True)

    file_count = 0
    done = False
    for root, dirs, files in os.walk(args.input_dir):
        root = pathlib.Path(root)
        for f in files:
            file_path = root / f
            file_ext = file_path.suffix
            file_ext = file_ext.lower() # Make it lowercase!

            if file_ext == "":
                file_ext = ".NONE"
            
            if file_ext not in extensions: # New extension
                extensions.add(file_ext)
                extension_folders[file_ext] = args.output_dir / file_ext[1:]
                extension_folders[file_ext].mkdir(parents=True, exist_ok=True)

            new_file_path = extension_folders[file_ext] / "{:07d}{}".format(
                    len(extension_mapping[file_ext]), file_ext)
            shutil.copy(file_path, new_file_path)
            extension_mapping[file_ext][str(new_file_path)] = str(file_path)

            file_count += 1
            if (args.count and file_count >= args.count):
                done = True
                break

        if done:
            break


    for e in extensions:
        f = open(extension_folders[e] / "filenames.json", "w")
        json.dump(extension_mapping[e], f, sort_keys=True, indent=4)
        f.close()


if __name__ == "__main__":
    main()
