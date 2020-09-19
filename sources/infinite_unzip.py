#!/usr/bin/env python3

import zipfile
import multiprocessing as mp
import subprocess
import argparse
import pathlib
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=pathlib.Path, 
        help="directory to list entries of")
args = parser.parse_args()

def get_zip_paths():
    zips = set()
    for root, dirs, files in os.walk(args.directory):
        for f in files:
            if f.endswith(".zip"):
                zips.add(pathlib.Path(root) / f)
    return zips


def unzip(p):
    try:
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(p.parent / p.stem)
    except:
        pass

old_paths = set()
while True:
    all_paths = get_zip_paths()
    new_paths = all_paths - old_paths
    if len(new_paths) == 0:
        break 

    # Yay subprocess
    #  subprocess.run("find " + str(args.directory) + 
    #          " -name \"*.zip\" | xargs -P 24 -I fileName sh -c 'unzip -o -d \"$(dirname \"fileName\")/$(basename -s .zip \"fileName\")\" \"fileName\"'", 
    #          shell=True)

    with mp.Pool(24) as pool:
        pool.map(unzip, new_paths)


    #  for p in new_paths:
    #      print(p)

    old_paths = all_paths
