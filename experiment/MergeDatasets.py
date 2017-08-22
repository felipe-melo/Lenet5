from theano import config, shared
import numpy as np
import theano.tensor as T
import os
from six.moves import cPickle as pickle
from util.Constants import Constants


def load_dataset():
    mnist_pickle_file = os.path.join(Constants.dataset_root, 'MNIST.pickle')

    fileObj = open(mnist_pickle_file, 'rb')
    mnistDatabase = pickle.load(fileObj)

    mnist_train_dataset = mnistDatabase['train_dataset']
    mnist_train_labels = mnistDatabase['train_labels'] + 10
    mnist_test_dataset = mnistDatabase['test_dataset']
    mnist_test_labels = mnistDatabase['test_labels'] + 10

    fileObj.close()

    notMNIST_pickle_file = os.path.join(Constants.dataset_root, 'notMNIST.pickle')

    fileObj = open(notMNIST_pickle_file, 'rb')

    notMNISTDatabase = pickle.load(fileObj)

    notmnist_train_dataset = notMNISTDatabase['train_dataset']
    notmnist_train_labels = notMNISTDatabase['train_labels']
    notmnist_test_dataset = notMNISTDatabase['test_dataset']
    notmnist_test_labels = notMNISTDatabase['test_labels']

    fileObj.close()

    notmnist_train_labels = notmnist_train_labels.reshape(notmnist_train_labels.shape[0], 1)
    notmnist_test_labels = notmnist_test_labels.reshape(notmnist_test_labels.shape[0], 1)

    x_1 = notmnist_train_dataset.shape[0] + mnist_train_dataset.shape[0]
    y_1 = notmnist_train_dataset.shape[1]

    x_2 = notmnist_test_dataset.shape[0] + mnist_test_dataset.shape[0]
    y_2 = notmnist_test_dataset.shape[1]

    a = notmnist_train_labels.shape[0] + mnist_train_labels.shape[0]
    b = notmnist_test_labels.shape[0] + mnist_test_labels.shape[0]

    train_dataset = np.zeros(shape=(x_1, y_1), dtype=config.floatX)
    test_dataset = np.zeros(shape=(x_2, y_2), dtype=config.floatX)
    train_labels = np.zeros(shape=(a, 1), dtype=config.floatX)
    test_labels = np.zeros(shape=(b, 1), dtype=config.floatX)

    ratio = int(notmnist_train_dataset.shape[0] / mnist_train_dataset.shape[0])

    j = 0

    for i in range(0, notmnist_train_dataset.shape[0], ratio):
        train_dataset[i:i + ratio] = notmnist_train_dataset[i:i + ratio]
        train_labels[i:i + ratio] = notmnist_train_labels[i:i + ratio]

        train_dataset[i + ratio:1] = mnist_train_dataset[j:1]
        train_labels[i + ratio:1] = mnist_train_labels[j:1]

    ratio = int(notmnist_test_dataset.shape[0] / mnist_test_dataset.shape[0])

    j = 0

    for i in range(0, notmnist_test_labels.shape[0], ratio):
        test_labels[i:i + ratio] = notmnist_test_labels[i:i + ratio]
        test_dataset[i:i + ratio] = notmnist_test_dataset[i:i + ratio]

        test_dataset[i + ratio:1] = mnist_test_dataset[j:1]
        test_labels[i + ratio:1] = mnist_test_labels[j:1]

        j += 1

    '''train_dataset = np.concatenate((train_dataset, _train_dataset), axis=0)
    test_dataset = np.concatenate((test_dataset, _test_dataset), axis=0)
    train_labels = np.concatenate((train_labels, _train_labels), axis=0)
    test_labels = np.concatenate((test_labels, _test_labels), axis=0)'''

    train_x = shared(np.asarray(train_dataset, dtype=config.floatX), borrow=True)
    train_y = shared(np.asarray(train_labels, dtype=config.floatX), borrow=True)
    test_x = shared(np.asarray(test_dataset, dtype=config.floatX), borrow=True)
    test_y = shared(np.asarray(test_labels, dtype=config.floatX), borrow=True)

    test_y = test_y.flatten()
    train_y = train_y.flatten()

    test_y = T.cast(test_y, 'int32')
    train_y = T.cast(train_y, 'int32')

    return train_x, train_y, test_x, test_y
