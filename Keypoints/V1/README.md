# Keypoint Prediction

## Train (CUDA env)

```
docker compose up
```

## conda (CPU env)

```bash
conda env create --name keypoints_v1 --file=environment.yaml
conda activate keypoints_v1
```

Or better use `micormamba`.

## nix (CPU env)

For now only CPU env. You could use cuda by setting `config.cudaSupport = true;` in `flake.nix`. But this probably need several hours to compile the required packages with cuda support!
