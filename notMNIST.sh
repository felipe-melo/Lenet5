#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/
epochs=20

#GPU
PYTHONPATH=$path_root python3 notMNIST/__init__.py $epochs > outputs/notMNIST_gpu.out

#CPU
PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 notMNIST/__init__.py $epochs > outputs/notMNIST_cpu.out