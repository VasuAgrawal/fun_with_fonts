#!/usr/bin/env python3

for i, c in enumerate(range(ord(' '), ord('~')+1)):
    print(chr(c), end="")
    if (i+1) % 12 == 0:
        print()
