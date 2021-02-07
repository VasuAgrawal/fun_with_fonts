#!/usr/bin/env python3

import subprocess
import json
import collections
import sys

atlas = ""
for i, c in enumerate(range(ord(' '), ord('~')+1)):
    atlas += chr(c)
    if (i+1) % 12 == 0:
        atlas += "\n"

# This is what I did OCR on, so this is all we can be sure-ish will render well.
sentence = "0123456789! THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. the quick brown fox jumps over the lazy dog?"
sentence = sentence.replace(" ", "")

dim = 2
target_dpi = 300 # DPI as close as possible to the OCR

stats_counter = collections.Counter()

print("Padding,Point,DPI,Offset,Image Count,Out Of Image,Out Of Cell,Empty Cell")
for padding in range(dim // 4):
    point = 1
    best_dpi = 1e10
    best_point = 0
    while True:
        dpi = 72 * (dim - 2 * padding) / point
        if dpi.is_integer():
            #  print(f"Padding {padding}: {point} pt, {int(dpi)} dpi")
            if abs(dpi - target_dpi) < abs(best_dpi - target_dpi):
                best_dpi = dpi 
                best_point = point

        point += 1
        if dpi < 1:
            break

    if best_dpi < 1:
        continue

    prev_out = None
    lowered = False
    increased = False
    for offset in range(dim + 1):
        output = subprocess.run(
                ["../build/RenderAtlas",
                    "--font-dir", "/data/datasets/fonts/split_05/train",
                    "--output-dir",
                    f"/tmp/fonts/pad_{padding:02d}_pt_{best_point:04d}_dpi_{int(best_dpi):03d}_off_{offset:02d}",
                    "--count", "0",
                    "--dpi", str(int(best_dpi)),
                    "--point", str(best_point),
                    "--border", str(2*padding),
                    "--padding", str(padding),
                    "--offset", str(offset),
                    "--atlas", sentence,
                    "--stats",
                    "--nosave",
                    "--thread-count", "48",
                    ],
                capture_output = True
                )

        try:
            stats = json.loads(output.stdout.decode('utf-8'))
            print(",".join(map(str, [
                padding, best_point, int(best_dpi), offset, stats['total'],
                stats['out_of_image_bounds'], stats['out_of_cell_bounds'],
                stats['empty_cell']])))
        except:
            print(f"Error with padding {padding}, best_point {best_point}, best_dpi {best_dpi}, offset {offset}",
                    file=sys.stderr)
            print(f"stdout: {output.stdout.decode('utf-8')}", file=sys.stderr)
            continue

        #  # Don't need to do any more once they're all outside of bounds
        #  if stats['out_of_cell_bounds'] == stats['total']:
        #      break

        if prev_out is None:
            prev_out = stats['out_of_cell_bounds']

        if stats['out_of_cell_bounds'] < prev_out:
            lowered = True

        if stats['out_of_cell_bounds'] > prev_out:
            increased = True

        if (lowered == True or increased == True) and stats['out_of_cell_bounds'] == stats['total']:
            # We've reached the peak again
            break

        # Don't need to do any more once they're all empty
        if stats['empty_cell'] == stats['total']:
            break

        prev_out = stats['out_of_cell_bounds']

