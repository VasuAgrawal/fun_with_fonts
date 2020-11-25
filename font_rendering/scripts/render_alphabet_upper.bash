#!/bin/bash

if [ -z $1 ]; then
    echo "Enter output path"
    exit
fi

dpi=300
point=12
border=0
padding=0

for i in {A..Z}; do
    train_output="$1/train/$i"
    test_output="$1/test/$i"
    ../build/RenderAtlas --font-dir "/data/datasets/fonts/split_05/train" \
        --output-dir $train_output \
        --count 0 \
        --dpi $dpi \
        --point $point \
        --border $border \
        --padding $padding \
        --atlas "$i" &&

    ../build/RenderAtlas --font-dir "/data/datasets/fonts/split_05/test" \
        --output-dir $test_output \
        --count 0 \
        --dpi $dpi \
        --point $point \
        --border $border \
        --padding $padding \
        --atlas "$i" &&

    # Note that we're deleting duplicates across train and test, rather than
    # individually within each set.
    jdupes --delete --noprompt --recurse $train_output $test_output > /dev/null &
done

wait
