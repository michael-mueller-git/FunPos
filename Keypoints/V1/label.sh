#!/usr/bin/env bash

for f in ./data/videos/*.mkv; do
    echo $f
    nix develop --command python3 label.py $f
done
for f in ./data/videos/*.mp4; do
    echo $f
    nix develop --command python3 label.py $f
done
