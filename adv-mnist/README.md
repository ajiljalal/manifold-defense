# Adversarial Training Using the Overpowered Attack
This repository contains code to perform adversarial training using the overpowered attack proposed in our paper
[The Robust Manifold Defense: Adversarial Training Using Generative Models](https://arxiv.org/abs/1712.09196)

# Setup
---
1. To set up the necessary environment assuming you use Python3.5, please do:
```shell
$./setup.sh
```
If not, please edit ```setup.sh``` appropriately.

2. Once the environment is set up, please do
```shell
$ source env/bin/activate
```
---

# Training Baseline Madry Model
To train a baseline model similar to Madry et al., do
```shell
$ python adv_train.py --dataset mnist --mode l2 --eps 1.5 --validation-set --opt adam --sgd-lr 1e-4 --no-norm --save-str <string> --num-epochs <num-epochs> --save-iters <save-frequency> --random-step
```

This will train a classifier on the mnist dataset with white box PGD L2 perturbation with norm 1.5. The checkpoint with best validation set accuracy will be saved in ```results/``` as ```results/mnist_l2_<string>_best```. 

If you choose to save checkpoints at regular intervals, then the checkpoints will be saved as ```results/mnist_l2_<string>_<epoch>``` where ```<epoch>``` is an integer multiple of ```<save-frequency>```.

If you only want to save the best model, choose ```<save-frequency>``` to be greater than ```<num-epochs>```. 
