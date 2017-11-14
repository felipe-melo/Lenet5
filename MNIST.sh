#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/
epochs=30

#GPU Profile tempo e memoria
#PYTHONPATH=$path_root nvprof --print-gpu-trace -u s --log-file outputs/mnist_cuda_profile.out python3 MNIST/__init__.py $epochs > outputs/mnist_gpu.out

#GPU Profile flops
#PYTHONPATH=$path_root nvprof --metrics flop_count_sp,flop_sp_efficiency,flop_count_dp,flop_dp_efficiency --log-file outputs/mnist_flops.out python3 MNIST/__init__.py $epochs > outputs/mnist_gpu.out

#CPU
PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 MNIST/__init__.py $epochs > outputs/mnist_cpu.out
