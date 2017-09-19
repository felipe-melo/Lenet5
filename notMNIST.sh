#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/
epochs=1

#PYTHONPATH=$path_root nvprof --print-gpu-trace -u s --log-file outputs/notMNIST_gpu_time.out python3 notMNIST/__init__.py

PYTHONPATH=$path_root python3 notMNIST/__init__.py $epochs > outputs/notMNIST_gpu.out

PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 notMNIST/__init__.py $epochs > outputs/notMNIST_cpu.out