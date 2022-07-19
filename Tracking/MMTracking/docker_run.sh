#!/bin/bash

docker run -ti --rm -v $PWD:/src -v $PWD/hub:/root/.cache/torch mmtrack /bin/bash -c "cd /src && python main.py"
# docker run -ti --rm --gpus all -v $PWD:/src -v $PWD/hub:/root/.cache/torch/hub mmtrack /bin/bash -c "cd /src && python main.py"
