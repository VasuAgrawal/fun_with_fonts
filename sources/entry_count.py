#!/usr/bin/env python3

import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=pathlib.Path, 
        help="directory to list entries of")
args = parser.parse_args()

import os

counts = {}
for child in args.directory.iterdir():
    if child.is_dir():
        counts[str(child.name)] = len(os.listdir(child))

for path, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{path}: {count} elements")
        
