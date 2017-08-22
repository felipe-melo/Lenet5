import time

import theano.tensor as T
from lenet_5.ConvPoolLayer import ConvPoolLayer, np
from lenet_5.FullyConnected import FullyConnected
from theano import function

from lenet_5.HiddenLayer import HiddenLayer

rng = np.random


class Lenet5(object):

    def __init__(self, batch_size=150, labels=10):
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

    def run_train(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset=None, valid_labels=None,
                  epochs=25, eta=0.3):

        index = T.lscalar()

        n_train_batches = train_dataset.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batches = test_dataset.get_value(borrow=True).shape[0] // self.batch_size
        if valid_dataset is not None:
            n_valid_batches = valid_dataset.get_value(borrow=True).shape[0] // self.batch_size

        updates = [(param_i, param_i - eta * grad_i) for param_i, grad_i in zip(self.params, self.grads)]

        self.train = function(
            inputs=[index],
            outputs=[self.f4.confusion_matrix(self.y)],
            #outputs=[self.f4.output],
            updates=updates,
            givens={
                self.x: train_dataset[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: train_labels[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        if valid_dataset is not None:
            validate_model = function(
                inputs=[index],
                outputs=[self.f4.confusion_matrix(self.y)],
                givens={
                    self.x: valid_dataset[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: valid_labels[index * self.batch_size: (index + 1) * self.batch_size]
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

        print("trainning...")
        t1 = time.time()

        epoch = 0
        while epoch < epochs:
            epoch += 1

            confucion_matrix = np.zeros((self.labels, self.labels), dtype='int')

            for mini_batch_index in range(n_train_batches):
                confu = self.train(mini_batch_index)
                print(confu)
                confucion_matrix += confu[0]

            print("epocha:", epoch, "accuracy:", confucion_matrix.diagonal().sum() / confucion_matrix.sum())

        print("trainning time:", time.time() - t1)

        print("testing...")
        t1 = time.time()
        confucion_matrix = np.zeros((self.labels, self.labels), dtype='int')

        for i in range(n_test_batches):
            confu = self.test_model(i)
            confucion_matrix += confu[0]

        test_accuracy = confucion_matrix.diagonal().sum() / confucion_matrix.sum()

        for i in range(self.labels):
            precision = confucion_matrix[i, i] / (confucion_matrix[:, i].sum())
            recall = confucion_matrix[i, i] / (confucion_matrix[i, :].sum())
            f1_score = 2 * (precision * recall) / (precision + recall)
            print("Precision class:", i, ":", precision)
            print("Recall class:", i, ":", recall)
            print("F1 score class:", i, ":", f1_score)

        print("Epocha:", epoch, "test accuracy:", test_accuracy * 100, "%")
        print("testing time", time.time() - t1)
