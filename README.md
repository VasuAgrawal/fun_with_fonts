First, download a bunch of font files and put them into the `sources` folder.
The included script will download a bunch of fonts from `dafont.com` over the
course of a day or two. Maybe you want to add some more.

Then, run the `copy_by_extension.sh` script which will generate a `by_extension`
folder containing fonts sorted by their file extension. Before doing this, you
may need to fix the permissions in your sources folder, otherwise you may get
errors saying a few files can't be copied.

```
find sources -type f -exec chmod 644 {} \;
find sources -type d -exec chmod 755 {} \;
```

Then, you'll want to build the font renderer:

```
cd font_rendering
./build.sh
```

which then generates some binaries in a `build` folder inside there. You want to
then run the following commands:

```
./build/third --font-dir ../by_extension/ttf --output-dir ../rendered/ttf
./build/third --font-dir ../by_extension/otf --output-dir ../rendered/otf
./build/third --font-dir ../by_extension/svg --output-dir ../rendered/svg
./build/third --font-dir ../by_extension/eot --output-dir ../rendered/eot
./build/third --font-dir ../by_extension/woff --output-dir ../rendered/woff
./build/third --font-dir ../by_extension/woff2 --output-dir ../rendered/woff2
```

The important ones are ttf / otf. The others are more there as a curiosity.
Other font types can probably be supported. The reason we separate them by
extension is because they'll override eachother otherwise, and there's slight
differences in how a ttf vs otf are rendered (apparently) that I don't want to
randomly introduce into the data.

## Building 

Install the dependencies:

1. C++20 capable compiler
1. [libfreetype](https://www.freetype.org/index.html)
1. [opencv](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
1. [libfmt](https://fmt.dev/latest/usage.html)
1. [gflags](https://gflags.github.io/gflags/#download)
1. [google benchmark](https://github.com/google/benchmark#installation)

Then, in standard `cmake` fashion:

```
cd font_rendering
mkdir build
cd build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 48
```

## Benchmarking

I wrote a couple of benchmarks to test performance.

### Image Writing Benchmark

This one's meant to evaluate different `cv::imwrite` parameter combinations to
help determine an optimal point in the output size / compression time tradeoff.
Try the following:

```
./build/ImageWriteBenchmark --image_path [image.png] --rgb_image=false
```

Note that all of the standard google benchmark flags are also supported (even if
they're not listed). For example, to save the benchmarks:

```
./build/ImageWriteBenchmark --image_path [image.png] --rgb_image=false \
    --benchmark_format=csv > imwrite_benchmarks.csv
```

### Font Renderer Benchmark

Just how fast can I render things? Let's find some bottlenecks.
