# FunPos

Deep Learning Model to generate funscripts.

This repository contains my test scripts to evaluate the automatically generation of funscripts with deep learning models.

To the standard question, I can say: I can not provide an pretrained Model! I do not have a sufficiently powerful GPU to train such a model at all.

## TODO

- [ ] Implement prediction for middle frame not last frame
- [ ] implement batch > 1 in dataloader:
  - Currently not required since training running at 100% on my Hardware
- [ ] increase model layers
  - Not possible i have only 6GB Video Ram 😥.
- [ ] Implement validation to prevent over fitting
