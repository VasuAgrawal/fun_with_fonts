#!/bin/bash

if [ -z $1 ]; then
    echo "Enter output path"
    exit
fi

for i in {0..9}; do
    output="$1/$i"
    ../build/RenderAtlas --font-dir /data/datasets/fonts/by_extension \
        --output-dir $output \
        --count 0 \
        --dpi 150 \
        --point 12 \
        --border 0 \
        --padding 0 \
        --atlas "$i"

    jdupes --delete --noprompt --recurse $output > /dev/null
done

