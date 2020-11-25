#!/usr/bin/env python3

import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--input_dir", type=pathlib.Path, required=True,
        help="Directory to crawl for files")
parser.add_argument("-o", "--output_dir", type=pathlib.Path,
        default="by_extension")

parser.add_argument("-c", "--count", type=int, default=0,
        help="Stop after copying this many files (0 means don't stop)",
        )
parser.add_argument("-s", "--split", type=float, default=.1,
        help="Ratio of data to split off as test")
args = parser.parse_args()

import os
import random
import shutil

def main():
    paths = []
    for root, dirs, files in os.walk(args.input_dir):
        root = pathlib.Path(root)
        for f in files:
            file_path = root / f
            paths.append(file_path)

    random.seed(42)
    random.shuffle(paths)
    if args.count > 0:
        paths = paths[:args.count]
   
    split = int(len(paths) * args.split)
    
    test_dir = args.output_dir/ "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[:split]: # Test data first
        shutil.copy(path, test_dir / path.name)

    train_dir = args.output_dir/ "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[split:]: # Train data next
        shutil.copy(path, train_dir / path.name)

if __name__ == "__main__":
    main()
