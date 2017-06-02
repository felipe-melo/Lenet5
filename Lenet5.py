from theano import function
import theano.tensor as T
from ConvPoolLayer import ConvPoolLayer, np
from FullyConnected import FullyConnected
from HiddenLayer import HiddenLayer

rng = np.random


class Lenet5(object):

    def __init__(self, batch_size=150):
        self.x = T.matrix('x')
        self.y = T.ivector('y')

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
            n_out=10
        )

        self.cost = -T.mean(T.log(self.f4.output)[T.arange(self.y.shape[0]), self.y])

        self.params = self.convPool1.params + self.convPool2.params + self.h3.params + self.f4.params
        self.grads = T.grad(self.cost, self.params)

        self.train = None
        self.test_model = None

    def run_train(self, database, epochs=50, eta=0.1):

        index = T.lscalar()

        train_set_x, train_set_y = database[0]
        test_set_x, test_set_y = database[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // self.batch_size

        updates = [(param_i, param_i - eta * grad_i) for param_i, grad_i in zip(self.params, self.grads)]

        self.train = function(
            inputs=[index],
            outputs=[self.f4.errors(self.y)],
            updates=updates,
            givens={
                self.x: train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        self.test_model = function(
            inputs=[index],
            outputs=[self.f4.errors(self.y)],
            givens={
                self.x: test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: test_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        epoch = 0
        while epoch < epochs:
            epoch += 1
            errors = []
            for mini_batch_index in range(n_train_batches):
                error = self.train(mini_batch_index)
                errors.append(error)
                #iter = (epoch - 1) * n_train_batches + mini_batch_index

                #if (iter + 1) % self.frequency == 0:
            #test_losses = [self.test_model(i) for i in range(n_test_batches)]
            #test_score = np.mean(test_losses)

            #print("Época: ", epoch, " testError: ", test_score * 100, "%")
            print("Época: ", epoch, " trainError: ", np.mean(errors) * 100, "%")
