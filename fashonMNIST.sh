#!/bin/sh

path_root=~/Documentos/Felipe/Lenet5/
epochs=30

#GPU Profile tempo e memoria
PYTHONPATH=$path_root nvprof --print-gpu-trace -u s --log-file outputs/fashonmnist_cuda_profile.out python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_gpu.json

#CPU
#PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 fashonMNIST/__init__.py $epochs > outputs/fashon_mnist_cpu.json
