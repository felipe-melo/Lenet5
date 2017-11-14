#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/
epochs=20

#GPU
PYTHONPATH=$path_root python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_gpu.out

#CPU
PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_cpu.out