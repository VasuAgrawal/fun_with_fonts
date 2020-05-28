#!/bin/bash

cd font_rendering
./build.sh
cd -

rm -rf rendered # not going to do this one with user input by accident lmao

output_dir=rendered
input_dir=by_extension
file_extensions=`ls $input_dir`

for ext in ${file_extensions[@]}; do
    echo $ext

    ./font_rendering/build/third \
        --font-dir by_extension/$ext \
        --output_dir $output_dir/$ext \
        --thread_count 48

done

