#!/bin/bash

out_dir=by_extension
mkdir -p $out_dir

file_extensions=(ttf otf eot woff woff2 svg)
for ext in ${file_extensions[@]}; do
    ext_out="${out_dir}/${ext}"
    echo "Writing to ${ext_out}"
    mkdir -p $ext_out

    find sources -iname "*.${ext}" -type f -exec cp {} $ext_out \;

done
