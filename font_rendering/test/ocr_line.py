#!/usr/bin/env python3

# Print all the things that have been validated by OCR, in a line.

for i, c in enumerate(range(ord('A'), ord('Z')+1)):
    print(chr(c), end="")
#  print()

for i, c in enumerate(range(ord('a'), ord('z')+1)):
    print(chr(c), end="")
#  print()

print("!?.")
