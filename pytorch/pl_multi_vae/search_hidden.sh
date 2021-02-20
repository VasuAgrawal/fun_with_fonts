#!/bin/bash

for i in {1..64}; do
    python3 ./main.py \
        --comment hidden_search \
        --channels 55 \
        --hidden $i \
        --gpus 2 \
        --accelerator ddp \
        --precision 16 \
        --deterministic true \
        --terminate_on_nan true \
        --benchmark true \
        --max_epochs 15
done
