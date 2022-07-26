# MMTracking

## Setup

```bash
bash ./setup.sh
conda env create --name MMTracking --file=environment.yml
conda activate MMTracking
pip install torch==1.8.0+cu111 torchvision=0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall mmcv-full
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

```

To update the env use:

```bash
conda env update --name MMTracking --file=environment.yml
```

## Run

```bash
conda activate MMTracking
python -s main.py
```

NOTE: `-s` ensure not to add `.local/lib/python3.X/site-packages` to `sys.path` (see [#7173](https://github.com/conda/conda/issues/7173))
