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
