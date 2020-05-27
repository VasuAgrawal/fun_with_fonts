#!/bin/sh

rm -rf build
mkdir -p build

clang++ once.cpp --std=c++20 \
    `pkg-config --cflags --libs freetype2 fmt gflags opencv4` -lglog \
    -Wno-deprecated-anon-enum-enum-conversion \
    -o build/once -O3 &

clang++ third.cpp  \
    `pkg-config --cflags --libs freetype2 fmt gflags opencv4 libfolly` -lglog \
    -Wno-deprecated-anon-enum-enum-conversion \
    --std=c++20 \
    -o build/third -g -O0 &

clang++ second.cpp --std=c++20 \
    `pkg-config --cflags --libs freetype2 fmt gflags opencv4` -lglog \
    -Wno-deprecated-anon-enum-enum-conversion \
    -o build/second -O2 &

clang++ first.cpp --std=c++20 \
    `pkg-config --cflags --libs freetype2 fmt gflags opencv4` \
    -Wno-deprecated-anon-enum-enum-conversion \
    -o build/first &

clang++ render_bench.cpp --std=c++20 \
    `pkg-config --cflags --libs freetype2 fmt opencv4 benchmark` -lpthread -lglog\
    -o build/render_bench -O3 &

wait
