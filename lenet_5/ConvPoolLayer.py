import numpy as np
from theano.tensor.nnet import conv2d
from theano import shared, config
from theano.tensor.nnet import sigmoid


class ConvPoolLayer(object):

    def __init__(self, rng, filter_shape, image_shape, pool_size, activation_fn=sigmoid):

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn = activation_fn

        #fan_in = np.prod(filter_shape[1:])
        #fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) / np.prod((2, 2))
        #w_bound = np.sqrt(6. / (fan_in + fan_out))

        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))

        self.W = shared(
            np.array(
                rng.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=config.floatX),
            borrow=True)

        self.bias = shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=config.floatX),
            borrow=True)

        self.output = None

        self.params = [self.W, self.bias]

    def run_layer(self, inpt):
        conv_output = conv2d(input=inpt, filters=self.W, filter_shape=self.filter_shape, input_shape=self.image_shape, subsample=self.pool_size)

        #pooled_out = pool.pool_2d(input=conv_output, ws=self.pool_size, mode='sum', ignore_border=True)

        self.output = self.activation_fn(conv_output+ self.bias.dimshuffle('x', 0, 'x', 'x'))
