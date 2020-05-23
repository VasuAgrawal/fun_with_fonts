#!/bin/sh

mkdir -p build
clang++ first.cpp --std=c++20 \
    `pkg-config --cflags --libs freetype2 fmt gflags opencv4` \
    -Wno-deprecated-anon-enum-enum-conversion \
    -o build/first

