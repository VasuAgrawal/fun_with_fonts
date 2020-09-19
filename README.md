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

## Round 2

1. Download a bunch of files. Extract all the non-zip files by hand. Use the
   following script as a convenience to continually, recursively extract zip
   files:

    ```
    ./sources/infinite_unzip.py /data/sources
    ```

1. Group all the files by extension:

    ```
    ./copy_and_rename_by_ext.py -d /data/sources -o /data/by_extension
    ```

1. Remove duplicates from the grouped extensions folder, and log which ones are
   duplicates.

    ```
    jdupes --delete --noprompt --recurse /data/by_extension > /data/jdupes.txt 
    ```

1. Figure out which of your extensions are actually renderable. This is mostly
   to deal with low quality data sources:

    ```
    for d in $(ls /data/by_extension); do
        if [ -d "/data/by_extension/$d" ]; then
            echo $d;
            ./build/ShowAtlas --font-dir "/data/by_extension/$d" --atlas "$(./test/ascii_atlas.py )" > /dev/null; 
        fi;
    done
    ```

   As that's running, you should see rendered things come up. Look at the most
   recent output in command line to get the folder extension. For my data, the
   following folders were useful. I've also listed the size of each, found by
   using the `./entry_count.py` script.

    1: 7
    bin: 19
    dfont: 117
    eot: 146
    html: 1392
    mrf: 4
    mtt: 2
    NONE: 40694
    otf: 63725
    pfa: 7
    pfb: 29109
    pfm: 31909
    ps: 3
    suit: 3018
    suit$: 2
    ttc: 341
    ttf: 216438
    txt: 16475
    woff: 564
    woff2: 181

1. Filter out bad files - these are files that can't be loaded, or fonts with
   characters that fail to load, or fonts with characters that are empty for the
   printable ASCII characters (' ' to '~'), excluding ' '.

    ```
    ./build/MoveBadFonts --font-dir /data/by_extension --thread-count 48 --error_dir /data/by_extension_errors
    ```

   This seems to like to segfault (since it's trying to open a bunch of random
   filetypes as fonts), so just keep running it until it completes successfully
   and you get 0s across the board in the return values.

   Now, all the fonts that remain in `/data/by_extension` should be renderable
   without any errors. Here's what remains for me:

    ```
    ttf: 174563 elements
    otf: 50593 elements
    pfb: 23665 elements
    woff: 431 elements
    ttc: 332 elements
    dfont: 114 elements
    woff2: 111 elements
    NONE: 25 elements
    pfm: 22 elements
    pfa: 5 elements
    suit: 4 elements
    mrf: 1 elements
    ps: 1 elements
    1: 1 elements
    suit$: 1 elements
    eot: 1 elements
    mtt: 1 elements
    html: 1 elements
    ```

1. Now, we filter out the fonts that have duplicate characters in the wrong
   places. We end up with 3 buckets - fonts that have no duplicate characters at
   all, fonts that have some duplicates (acceptable ones, such as uppercase and
   lowercase letters being the same), and unacceptable duplicates (such as A and
   B being the same). The first bucket is what's left in the original directory,
   with the other two buckets being added to the error dir.

   The whitelist was painstakingly manually constructed based on output from the
   IdentifyDuplicates binary.

    ```
    ./build/MoveDuplicates --font-dir /data/by_extension --error-dir /data/by_extension_errors --whitelist full_whitelist.txt
    ```

   Here's what remains after this filter step:

    ```
    ttf: 110754 elements
    otf: 38911 elements
    pfb: 20605 elements
    ttc: 266 elements
    woff: 251 elements
    dfont: 107 elements
    woff2: 52 elements
    pfm: 17 elements
    NONE: 13 elements
    pfa: 5 elements
    suit: 3 elements
    ps: 1 elements
    1: 1 elements
    eot: 1 elements
    ```

   Errors:

    ```
    failed_duplicates: 46265 elements
    allowed_duplicates: 32620 elements
    ```

1. Finally, we filter out fonts that don't meet an OCR test with tesseract. The
   initial version we're doing here is very strict, intended to extract the
   highest quality of data possible (automatically). We prefer to have false
   negatives than false positives. The OCR test tries to render a pangram with
   common symbols, perform OCR on it, and then check if the Levenshtein distance
   to the computed string is 0. Any nonzero distances are categorized as such in
   the output errors folder.

   Note that this is expected to take a while - it took a few hours on my 24
   core (48 thread) CPU, running at near 100% utilization the entire time.
   Still, better than annotating all the data by hand, right?

    ```
    ./build/MoveByOcr --font-dir /data/by_extension --error-dir /data/by_extension_errors --atlas "0123456789! THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. the quick brown fox jumps over the lazy dog?" --count 0 --thread_count 36 --csv --lowercase=false --dpi 300 --point 12 --padding 100 
    ```

1. At this point, you might as well as clean up empty directories:

    ```
    find /data/by_extension -empty -type d -delete
    ```
