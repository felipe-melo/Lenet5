#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/
epochs=1

#MNIST cpu
#PYTHONPATH=$path_root THEANO_FLAGS='device=gpu' nvprof --metrics flop_count_sp,flop_sp_efficiency,flop_count_dp,flop_dp_efficiency --log-file outputs/mnist_cpu_flop.out python3 MNIST/__init__.py

PYTHONPATH=$path_root python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_gpu.out

PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_cpu.out