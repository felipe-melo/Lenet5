import numpy as np
from theano.tensor.signal import pool
from theano import shared, config
import theano.tensor as T


class Subsampling(object):

    def __init__(self, rng, _input, subsampling_size, filter_shape):

        #preciso ver como iniciar os valores do bias

        b_value = np.zeros((filter_shape[0],), dtype=config.floatX)
        self.bias = shared(value=b_value, borrow=True)

        self.coefficient = shared(np.float32(rng.rand()), borrow=True)

        pooled_out = pool.pool_2d(input=_input, ws=subsampling_size, mode='sum', ignore_border=False)

        self.output = T.tanh(pooled_out * self.coefficient + self.bias.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.bias, self.coefficient]
