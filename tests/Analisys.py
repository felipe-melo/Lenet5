#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import function, config, shared
import theano.tensor as T
import numpy as np
import os

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #Recomendado quando profile=True, deixa mais lento porém é mais preciso
#config.profile = True
#config.profile_memory=False
#config.profile_optimizer=True
#config.debugprint = True


def dot_matrix(matrix1, matrix2):
    if matrix1.shape[1] != matrix2.shape[0]:
        raise Exception('Matrizes não estão alinhadas')

    x = T.matrix('x')
    y = T.matrix('y')

    z = T.dot(x, y)

    f = function(inputs=[x, y], outputs=z)

    f(matrix1, matrix2)


def dot_matrix_share(matrix1, matrix2, times):
    if matrix1.shape[1] != matrix2.shape[0]:
        raise Exception('Matrizes não estão alinhadas')

    x = shared(matrix1, name="x", borrow=True)
    y = shared(matrix2, name="y", borrow=True)

    z = T.dot(x, y)

    f = function(inputs=[], outputs=z)

    for i in range(times):
        f()


def add_matrix(matrix1, matrix2):
    x = T.matrix('x')
    y = T.matrix('y')

    z = T.add(x, y)

    f = function(inputs=[x, y], outputs=z)

    f(matrix1, matrix2)


def add_matrix_share(matrix1, matrix2):

    x = shared(matrix1, name="x", borrow=True)
    y = shared(matrix2, name="y", borrow=True)

    z = T.add(x, y)

    f = function(inputs=[], outputs=z)

    f()


def multiply_matrix(matrix1, matrix2):
    x = T.matrix('x')
    y = T.matrix('y')

    z = T.mul(x, y)

    f = function(inputs=[x, y], outputs=z)

    f(matrix1, matrix2)


def multiply_matrix_share(matrix1, matrix2):

    x = shared(matrix1, name="x", borrow=True)
    y = shared(matrix2, name="y", borrow=True)

    z = T.mul(x, y)

    f = function(inputs=[], outputs=z)

    f()


def sigmoid(matrix1):

    x = T.matrix('x')

    #z = 1 / (1 + T.exp(-x))
    z = T.nnet.nnet.sigmoid(x)

    f = function(inputs=[x], outputs=z)

    f(matrix1)


def sigmoid_share(matrix1):

    x = shared(matrix1, name="x", borrow=True)

    #z = 1 / (1 + T.exp(-x))
    z = T.nnet.nnet.sigmoid(x)

    f = function(inputs=[], outputs=z)

    f()


def tanh(matrix1):

    x = T.matrix('x')

    z = T.tanh(x)

    f = function(inputs=[x], outputs=z)

    f(matrix1)


def tanh_share(matrix1):

    x = shared(matrix1, name="x", borrow=True)

    z = T.tanh(x)

    f = function(inputs=[], outputs=z)

    f()


def scalar_matrix(scalar_num, matrix):

    x = T.matrix('x')
    s = T.scalar('s')

    z = s * x

    f = function(inputs=[x, s], outputs=z)

    f(matrix, scalar_num)


def scalar_matrix_share(scalar_num, matrix):

    x = shared(matrix, name="x", borrow=True)
    s = shared(np.float32(scalar_num), name="s", borrow=True)

    z = s * x

    f = function(inputs=[], outputs=z)

    f()


def softMax(matrix1):

    x = T.matrix('x')

    softm = T.nnet.softmax(x)

    f = function(inputs=[x], outputs=softm)

    f(matrix1)


def softMax_share(matrix):

    x = shared(matrix, name="x", borrow=True)

    softm = T.nnet.softmax(x)

    f = function(inputs=[], outputs=softm)

    f()
