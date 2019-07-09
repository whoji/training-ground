Largely based on the official pytorch tutorial on
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Some difference to the above tutorial:
* Put the various components into modules to avoid a large script
* Added a learning rate hyperparameter
* Better plot and visualization
* better terminal output
* [TODO] stack more frames for better training performance
* [TODO] save and load model.

More to do:
* [DONE] test if runnable on GPU
* [TODO] test on multiple platform
* [TODO] try to avoid using torchvision
* [TODO] better coding (between modules)
