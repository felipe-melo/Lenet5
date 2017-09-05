#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/

#PYTHONPATH=$path_root nvprof --print-gpu-trace -u s --log-file outputs/notMNIST_gpu_time.out python3 notMNIST/__init__.py

PYTHONPATH=$path_root python3 notMNIST/__init__.py > outputs/notMNIST_gpu.out

PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 notMNIST/__init__.py > outputs/notMNIST_cpu.out

#PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' nvprof --print-gpu-trace -u s --log-file outputs/MNIST_gpu_time.out python3 MNIST/__init__.py