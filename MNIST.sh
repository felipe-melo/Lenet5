#!/bin/sh

epochs=1

#MNIST cpu
#PYTHONPATH=$path_root THEANO_FLAGS='device=gpu' nvprof --metrics flop_count_sp,flop_sp_efficiency,flop_count_dp,flop_dp_efficiency --log-file outputs/mnist_cpu_flop.out python3 MNIST/__init__.py

PYTHONPATH=$path_root python3 init/__MNIST__.py $epochs > outputs/mnist_gpu.out

PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 MNIST/__init__.py $epochs > outputs/mnist_cpu.out
