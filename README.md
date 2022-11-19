# mini-pytorch
(Mini) pytorch framework from scratch

This repo contains a small, from scratch, modular implementation of pytorch autograd, module management and several standard modules (Linear, ReLU, Sigmoid, Conv2d, TransposeConv2d, MSE). In fact, all you need to do to use this framework is to remove `torch.nn` references from your pytorch script and voilà! `model.py` shows an example of how to use modules from this framework applied to denoising task.
