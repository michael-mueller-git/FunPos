# FunPos

Deep Learning Model to generate funscripts.

This repository contains my test scripts to evaluate the automatically generation of funscripts with deep learning models.

I put my test models on the [GitHub release page](https://github.com/michael-mueller-git/FunPos/releases), mostly for backup purpose.

## How to train the Model

1. Put some short video clips (20-30 senconds) to `./data/train`.
2. Use [OpenFunscripter](https://github.com/OpenFunscripter/OFS) and create funscripts for each video clip in `./data/train`.
3. Use `make label` to create the necessary labels for training
4. Use `make train` to train the model.
5. Use `make test` to test your trained model on a single video clip.

**Notes:** It may be necessary to make adjustments in the scripts!

## Open Points

- [ ] Implement prediction for middle frame not last frame
  - Maybe not necessary
- [ ] implement batch > 1 in dataloader:
  - Currently not required since training running at 100% on my Hardware
- [ ] increase model layers
  - Not possible i have only 6GB Video Ram.
- [ ] Implement validation to prevent over fitting
- [ ] Implement randomnes to data loader (for training) e.g. Gaussian noise, random tarnformation of frame sequences, ... to improve the generalization
- [ ] Add audio to improve prediction

## Current State

- Model 1 convergence when training, first results are ok
- Model 2 do not convergence when training
- Model 3 WIP
