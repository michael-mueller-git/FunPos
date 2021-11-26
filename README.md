# FunPos

Deep Learning Model to generate funscripts.

This repository contains my test scripts to evaluate the automatically generation of funscripts with deep learning models.

To the standard question, I can say: I can not provide an pretrained Model! I do not have a sufficiently powerful GPU to train such a model at all.

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

## Current State

- Model 1 convergence when training, firs results are good
- Model 2 do not convergence when training
- Model 3 WIP
