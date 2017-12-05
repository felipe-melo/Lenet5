import time

import theano.tensor as T
from lenet_5.ConvPoolLayer import ConvPoolLayer, np
from lenet_5.FullyConnected import FullyConnected
from theano import function

from lenet_5.HiddenLayer import HiddenLayer

rng = np.random


class Lenet5(object):

    def __init__(self, batch_size=200, labels=10):
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.labels = labels

        self.frequency = 100
        self.batch_size = batch_size
        image_shape = (batch_size, 1, 32, 32)

        layer0 = self.x.reshape(image_shape)

        filter_shape = (6, 1, 5, 5)
        pool_size = (2, 2)

        self.convPool1 = ConvPoolLayer(
            rng=rng,
            image_shape=image_shape,
            filter_shape=filter_shape,
            pool_size=pool_size
        )

        self.convPool1.run_layer(layer0)

        image_shape = (batch_size, 6, 14, 14)
        filter_shape = (16, 6, 5, 5)

        self.convPool2 = ConvPoolLayer(
            rng=rng,
            image_shape=image_shape,
            filter_shape=filter_shape,
            pool_size=pool_size
        )

        self.convPool2.run_layer(self.convPool1.output)

        self.h3 = HiddenLayer(
            rng,
            _input=self.convPool2.output.flatten(2),
            n_in=16*5*5,
            n_out=120
        )

        self.f4 = FullyConnected(
            rng,
            _input=self.h3.output,
            n_in=120,
            n_out=self.labels
        )

        self.cost = -T.mean(T.log(self.f4.output)[T.arange(self.y.shape[0]), self.y])

        self.params = self.convPool1.params + self.convPool2.params + self.h3.params + self.f4.params
        self.grads = T.grad(self.cost, self.params)

        self.train = None
        self.test_model = None

    def run_train(self, train_dataset, train_labels, test_dataset, test_labels, t0, epochs=25, eta=0.3):

        index = T.lscalar()

        n_train_batches = train_dataset.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batches = test_dataset.get_value(borrow=True).shape[0] // self.batch_size

        updates = [(param_i, param_i - eta * grad_i) for param_i, grad_i in zip(self.params, self.grads)]

        self.train = function(
            inputs=[index],
            outputs=[self.f4.confusion_matrix(self.y)],
            updates=updates,
            givens={
                self.x: train_dataset[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: train_labels[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        self.test_model = function(
            inputs=[index],
            outputs=[self.f4.confusion_matrix(self.y)],
            givens={
                self.x: test_dataset[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: test_labels[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        t1 = time.time()

        print('{"trainning":{ ', '"loading_time":', t1-t0, ', "epochs": [')

        epoch = 0
        while epoch < epochs:
            epoch += 1

            confucion_matrix = np.zeros((self.labels, self.labels), dtype='int')
            for mini_batch_index in range(n_train_batches):
                train_confu = self.train(mini_batch_index)
                confucion_matrix += train_confu[0]

            test_confucion_matrix = np.zeros((self.labels, self.labels), dtype='int')
            for mini_batch_index in range(n_test_batches):
                test_confu = self.test_model(mini_batch_index)
                test_confucion_matrix += test_confu[0]

            print('{"epoch"', ":{",
                  '"train_accuracy":', (confucion_matrix.diagonal().sum() / confucion_matrix.sum()),
                  ',"test_accuracy":', (test_confucion_matrix.diagonal().sum() / test_confucion_matrix.sum()),
                  ',"train_confusion_matrix": ', repr(confucion_matrix),
                  ',"test_confusion_matrix": ', repr(test_confucion_matrix))

            if epoch < epochs:
                print("}},")
            else:
                print("}}")

        print("],", '"time":', time.time() - t1, "}}")
