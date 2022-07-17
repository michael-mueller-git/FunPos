#!/bin/bash

echo "i recommend to use conda env not the docker image"
docker run -ti --rm --gpus all -v $PWD:/src -v $PWD/hub:/root/.cache/torch/hub vitpose /bin/bash -c "cd /src && python main.py"

