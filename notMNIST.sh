#!/bin/sh

path_root=~/Documentos/Felipe/Lenet5/
epochs=30

#GPU
PYTHONPATH=$path_root nvprof --print-gpu-trace -u s --log-file outputs/notMNIST_cuda_profile.out python3 notMNIST/__init__.py $epochs > outputs/notMNIST_gpu.json

#cpu
#PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 notMNIST/__init__.py $epochs > outputs/notMNIST_cpu.json
