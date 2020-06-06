#!/usr/bin/env python3

import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=pathlib.Path, required=True,
        help="Input directory")
args = parser.parse_args()

import cv2
import collections
import shutil
import time

KEEP_NAME = "keep"
DISCARD_NAME = "discard"
BACKLOG_SIZE = 500

class FilteredImage(object):
    def __init__(self, path):
        self._path = path
        self._im = cv2.imread(str(path))
        self._keep = None
        self._rendered = True


    def image(self):
        if not self._rendered:
            if self._keep == True:
                color = (0, 255, 0)
            elif self._keep == False:
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)

            cv2.rectangle(self._im, (0, 0), self._im.shape[:-1], color,  10)

            self._rendered = True
        return self._im


    def maybe_keep(self):
        if self._keep == None:
            self._keep = True
        self._rendered = False


    def keep(self):
        self._keep = True
        self._rendered = False


    def discard(self):
        self._keep = False
        self._rendered = False


    def reset(self):
        self._keep = None
        self._rendered = False


    def purge(self):
        if self._keep == True:
            new_file_path = self._path.parent / KEEP_NAME / self._path.name
            shutil.move(self._path, new_file_path)
        elif self._keep == False:
            new_file_path = self._path.parent / DISCARD_NAME / self._path.name
            shutil.move(self._path, new_file_path)


def main():

    (args.directory / KEEP_NAME).mkdir(exist_ok = True)
    (args.directory / DISCARD_NAME).mkdir(exist_ok = True)

    diriter = args.directory.iterdir()
    backlog = collections.deque()
    display_index = 0

    autoplay = False
    done = False
    while True:
        if display_index >= len(backlog):
            # add a new image
            while True:
                try:
                    path = next(diriter)
                    if path.suffix == ".png":
                        break
                except StopIteration:
                    done = True
                    break

            if done:
                break


            backlog.append(FilteredImage(path))

            if len(backlog) > BACKLOG_SIZE:
                backlog.popleft().purge()
                display_index -= 1


        cv2.imshow("Filter me", backlog[display_index].image())
        key = cv2.waitKey(150) & 0xFF
        if key == ord('q'):
            break
        elif key == 83: # right
            autoplay = False
            display_index += 1
        elif key == 81: # left
            autoplay = False
            display_index -= 1 
        elif key == 82: # up
            autoplay = False
            backlog[display_index].keep()
            display_index += 1
        elif key == 84: # up
            autoplay = False
            backlog[display_index].discard()
            display_index += 1
        elif key == ord('c'):
            autoplay = False
            backlog[display_index].reset()
            display_index += 1
        elif key == ord(' '):
            autoplay = not autoplay

        if key == 255: # No key pressed, just move on, same as up
            if (autoplay):
                backlog[display_index].maybe_keep()
                display_index += 1

        display_index = max(display_index, 0)


    for elem in backlog:
        elem.purge()

if __name__ == "__main__":
    main()
