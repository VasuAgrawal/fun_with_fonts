#!/bin/bash

if [ -z $1 ]; then
    echo "Enter output path"
    exit
fi

# size=128
# padding=8
# point=28
# dpi=288
# offset=27
# let border=2*$padding

# size=64
# padding=4
# point=14
# dpi=288
# offset=13
# let border=2*$padding

# size=32
# padding=2
# point=7
# dpi=288
# offset=7
# let border=2*$padding

size=16
padding=1
point=3
dpi=336
offset=3
let border=2*$padding

train_output="$1/$size/train"
validation_output="$1/$size/validation"
test_output="$1/$size/test"

../build/RenderAtlas --font-dir /data/datasets/fonts/split_05_val/train \
    --output-dir $train_output \
    --count 0 \
    --dpi $dpi \
    --point $point \
    --border $border \
    --padding $padding \
    --atlas "$(../test/ocr_line.py)" \
    --thread-count 42 &

../build/RenderAtlas --font-dir /data/datasets/fonts/split_05_val/validation \
    --output-dir $validation_output \
    --count 0 \
    --dpi $dpi \
    --point $point \
    --border $border \
    --padding $padding \
    --atlas "$(../test/ocr_line.py)" \
    --thread-count 3 &

../build/RenderAtlas --font-dir /data/datasets/fonts/split_05_val/test \
    --output-dir $test_output \
    --count 0 \
    --dpi $dpi \
    --point $point \
    --border $border \
    --padding $padding \
    --atlas "$(../test/ocr_line.py)" \
    --thread-count 3 &

wait

jdupes --delete --noprompt $train_output $validation_output $test_output > /dev/null
