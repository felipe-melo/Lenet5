#!/bin/sh

path_root=~/Desenvolvimento/ufrrj/TCC/Lenet5-TCC/

PYTHONPATH=$path_root THEANO_FLAGS='device=cpu' python3 tests/test.py