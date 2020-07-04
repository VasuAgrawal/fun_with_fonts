First, download a bunch of font files and put them into the `sources` folder.
The included script will download a bunch of fonts from `dafont.com` over the
course of a day or two. Maybe you want to add some more.

You may need to fix the permissions in your sources folder, otherwise you may
get errors saying a few files can't be copied.

```
$ sudo chown -R $USER:$USER sources
$ find sources -type f -exec chmod 644 {} \;
$ find sources -type d -exec chmod 755 {} \;
```

Then, run the `copy_and_rename_by_ext.py` script from the repo's root directory,
which will recursively traverse the specified folder (`sources`), pull out all
of the font files (based on extension matching), and put them into the
`by_extension` directory, with one folder per extension. 

```
./copy_and_rename_by_ext.py -d sources
```

You should then see something like the following:

```
$ tree --filelimit=10 by_extension 
by_extension
├── eot [132 entries exceeds filelimit, not opening dir]
├── otf [58836 entries exceeds filelimit, not opening dir]
├── svg [180 entries exceeds filelimit, not opening dir]
├── ttc [182 entries exceeds filelimit, not opening dir]
├── ttf [182653 entries exceeds filelimit, not opening dir]
├── woff [532 entries exceeds filelimit, not opening dir]
└── woff2 [175 entries exceeds filelimit, not opening dir]

7 directories, 0 files
```

Note also that there's a `filenames.json` file inside each of those folders
(e.g. `by_extension/ttf/filenames.json`) which holds a map from the renamed
filenames to the original filenames, in case you want to correspond them again.

Depending on the quality of your dataset, you may have a few, or many
duplicates. Processing duplicates slows down the rest of the pipeline, so we can
do a preprocessing step to remove them. From the `by_extension` directory, use
the `jdupes` program to filter out duplicates.

```
jdupes --delete --noprompt --recurse by_extension
```

Now, build the source. You might need to install some dependencies.

```
cd font_rendering
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 12
```

Now we get into some of the more opinionated filtering. From the large corpus of
fonts, we definitely want to remove the fonts which can't be loaded at all by
the `FreeType2` library. After that, we _probably_ want to remove fonts which
have characters that we can't load, or characters that don't draw anything to
the screen (except spaces, ` `). The question is how aggressively to do that.
I've chosen to filter on _every_ printable ASCII character, so everything from
` ` to `~`. If you want to be less aggressive, you can change the atlas
generation code in `font_rendering/src/move_bad_fonts.cpp`.

```
$ ./build/MoveBadFonts --font_dir ../by_extension --thread_count 48 \
    --error_dir ../by_extension_errors

Fonts with failed font loads: 1255
Fonts with failed char loads: 1567
Fonts with empty characters: 47335
```

Note that you can also use the `--dry_run` flag with `MoveBadFonts` to perform a
trial run and see how many fonts you're about to move.


You want to then run the following commands:

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
