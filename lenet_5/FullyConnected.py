import numpy as np
import theano.tensor as T
from theano import shared, config


class FullyConnected(object):

    def __init__(self, rng, _input, n_in, n_out):

        #preciso ver como iniciar os valores do weight
        w = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ), dtype=config.floatX)

        self.W = shared(value=w, name='W', borrow=True)

        self.b = shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_in), size=(n_out,)),
                dtype=config.floatX),
            borrow=True, name='b')

        self.output = T.nnet.softmax(T.dot(_input, self.W) + self.b)

        self.y_pred = T.argmax(self.output, axis=1)

        self.params = [self.W, self.b]

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

    def confusion_matrix(self, y):
        return T.nnet.confusion_matrix(y, self.y_pred)[0]
