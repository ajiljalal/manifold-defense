# The Robust Manifold Defense: Adversarial Training Using Generative Models

This repository provides code to reproduce results from the paper: [The Robust Manifold Defense: Adversarial Training Using Generative Models](https://arxiv.org/abs/1712.09196)

## Setup submodules
In this root directory, please run
```shell
$ git submodule update --init --recursive
$ git config --local submodule.recurse true
$ git config --local diff.submodule log
```
to update the necessary submodules.

The repository ```defense-gan-break/``` contains a break of DefenseGAN, while ```adv-mnist/``` contains code for adversarial training of using our overpowered attack on MNIST.

# Contributors
@[andrewilyas](https://github.com/andrewilyas) @[ajiljalal](https://github.com/ajiljalal)
