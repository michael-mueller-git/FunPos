# MMTracking

## Setup

```bash
conda env create --name MMTracking --file=environment.yml
```

## Run

```bash
conda activate MMTracking
python -s main.py
```

NOTE: `-s` ensure not to add `.local/lib/python3.X/site-packages` to `sys.path` (see [#7173](https://github.com/conda/conda/issues/7173))
