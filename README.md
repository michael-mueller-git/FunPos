# FunPos

Deep Learning Model to generate funscripts.

This repository contains my test scripts to evaluate the automatically generation of funscripts with deep learning models.

To the standard question, I can say: I can not provide an pretrained Model! I do not have a sufficiently powerful GPU to train such a model at all.

## How to train the Model

1. Put some short video clips (20-30 senconds) to `./data/train`.
2. Use OpenFunscripter and create funscripts for each video clip in `./data/train`.
3. Use `python create_01_parameter.py` to generate required training parameter (Select the Region of interest for each video and save them to the training directory).
4. Use `python create_02_label.py` to generate the regression labels for the training. We use the funscript and quadratic interpolate the position for each frame.
5. Use `python train.py` to train the model.
6. Use `python test.py` to test your trained model on a single video clip.

**Notes:** It may be necessary to make adjustments in the scripts!

## TODO

- [ ] Implement prediction for middle frame not last frame
- [ ] implement batch > 1 in dataloader:
  - Currently not required since training running at 100% on my Hardware
- [ ] increase model layers
  - Not possible i have only 6GB Video Ram 😥.
- [ ] Implement validation to prevent over fitting
