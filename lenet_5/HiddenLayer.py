import theano.tensor as T
from theano import shared, config
import numpy as np


class HiddenLayer(object):
    def __init__(self, rng, _input, n_in, n_out, activation=T.tanh):

        w_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=config.floatX
        )

        self.W = shared(value=w_values, name='W', borrow=True)

        self.b = shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_in), size=(n_out,)),
                dtype=config.floatX),
            borrow=True, name='b')

        lin_output = T.dot(_input, self.W) + self.b
        self.output = activation(lin_output)

        self.params = [self.W, self.b]
