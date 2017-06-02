#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
from tests.Analisys import *
import sys

rng = np.random.RandomState(22)

dimensions = [["512", "512"], ["512", "1024"], ["512", "2048"], ["512", "4096"], ["512", "8192"], ["512", "16384"],
              ["512", "32768"], ["512", "65536"], ["512", "131072"], ["512", "262144"], ["512", "524288"],
              ["512", "1048576"], ["512", "2097152"]]

'''
dimensions.append(["512", "1048576"]) #8GB
dimensions.append(["512", "2097152"]) #16GB
dimensions.append(["512", "4194304"]) #32GB
dimensions.append(["512", "8388608"]) #64GB
dimensions.append(["512", "16777216"]) #128GB
dimensions.append(["512", "33554432"]) #256GB

dimensions.append(["512", "67108864"]) #512GB
dimensions.append(["512", "134217728"]) #1TB
dimensions.append(["512", "268435456"]) #2TB
dimensions.append(["512", "536870912"]) #4TB
dimensions.append(["512", "1073741824"]) #8TB'''


def dot(lines, columns, vezes):

    m1 = get_matrix(lines, columns)
    m2 = get_matrix(columns, lines)

    start = time.time()
    dot_matrix_share(m1, m2, vezes)
    print(time.time() - start)


def get_matrix(lines, columns):
    return np.asmatrix(rng.rand(lines, columns), config.floatX)

if __name__ == "__main__":
    lines = 1024
    columns = 300
    times = 50

    dot(lines, columns, times)
