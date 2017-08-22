#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import struct
import numpy as np
from theano import config, shared
import theano.tensor as T
import os
from util.Constants import Constants
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle


train_images_file = Constants.dataset_root + "train-images.idx3-ubyte"
train_labels_file = Constants.dataset_root + "train-labels.idx1-ubyte"
test_images_file = Constants.dataset_root + "t10k-images.idx3-ubyte"
test_labels_file = Constants.dataset_root + "t10k-labels.idx1-ubyte"


def read_files(train_images=train_images_file, train_labels=train_labels_file, test_images=test_images_file,
               test_labels=test_labels_file):
    _train_x, _train_y = __read_original_file__(train_images, train_labels)
    _test_x, _test_y = __read_original_file__(test_images, test_labels)

    train_x = shared(np.asarray(_train_x, dtype=config.floatX), borrow=True)
    train_y = shared(np.asarray(_train_y, dtype=config.floatX), borrow=True)
    test_x = shared(np.asarray(_test_x, dtype=config.floatX), borrow=True)
    test_y = shared(np.asarray(_test_y, dtype=config.floatX), borrow=True)

    test_y = test_y.flatten()
    train_y = train_y.flatten()

    test_y = T.cast(test_y, 'int32')
    train_y = T.cast(train_y, 'int32')

    return [(train_x, train_y), (test_x, test_y)]


def __read_original_file__(image_x, label_y):
    img_file = open(image_x, 'rb').read()
    lab_file = open(label_y, 'rb').read()

    image_index = 0
    label_index = 0
    magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', img_file, image_index)
    magic, num_labels = struct.unpack_from('>II', lab_file, label_index)

    print('train_set:', num_images, ' label_set:', num_labels)

    data_x = np.ones((num_images, 1024))
    data_y = np.ones((num_labels, 1))

    image_index = struct.calcsize('>IIII')
    label_index = struct.calcsize('>II')

    for case in range(num_images):
        image = struct.unpack_from('>784B', img_file, image_index)
        label = struct.unpack_from('>1B', lab_file, label_index)
        image_index += struct.calcsize('>784B')
        label_index += struct.calcsize('>1B')
        image = np.array(image)
        image = image.reshape(28, 28)
        big_image = np.ones((32, 32)) * -0.1
        for i in range(28):
            for j in range(28):
                if image[i][j] > 0:
                    big_image[i + 2][j + 2] = 1.175
        big_image = big_image.reshape(1024)
        data_x[case] = np.asarray(big_image, dtype=config.floatX)
        data_y[case] = np.asarray(label[0], dtype=config.floatX)

    return data_x, data_y


def create_binary_files():
    _train_x, _train_y = __read_original_file__(train_images_file, train_labels_file)
    _test_x, _test_y = __read_original_file__(test_images_file, test_labels_file)

    pickle_file = os.path.join('database', 'MNIST.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': _train_x,
            'train_labels': _train_y,
            'test_dataset': _test_x,
            'test_labels': _test_y,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def plotingSomeImg():
    pickle_file = os.path.join('database', 'MNIST.pickle')

    fileObj = open(pickle_file, 'rb')

    allDatabase = pickle.load(fileObj)

    train_dataset = allDatabase['train_dataset']
    train_labels = allDatabase['train_labels']
    test_dataset = allDatabase['test_dataset']
    test_labels = allDatabase['test_labels']

    print('Training:', train_dataset.shape, 'Label:', train_labels.shape)
    print('Testing:', test_dataset.shape, 'Label:', test_labels.shape)

    for i in range(10):
        sample_idx = np.random.randint(train_dataset.shape[0])  # pick a random image index
        sample_image = train_dataset[sample_idx].reshape(32, 32)
        print(train_labels[sample_idx])
        plt.figure()
        plt.imshow(sample_image)
        plt.show()

    for i in range(10):
        sample_idx = np.random.randint(test_dataset.shape[0])  # pick a random image index
        sample_image = test_dataset[sample_idx].reshape(32, 32)
        print(test_labels[sample_idx])
        plt.figure()
        plt.imshow(sample_image)
        plt.show()


def load_dataset():
    pickle_file = os.path.join(Constants.dataset_root, 'MNIST.pickle')

    fileObj = open(pickle_file, 'rb')

    allDatabase = pickle.load(fileObj)

    train_dataset = allDatabase['train_dataset']
    train_labels = allDatabase['train_labels']
    test_dataset = allDatabase['test_dataset']
    test_labels = allDatabase['test_labels']

    fileObj.close()

    train_x = shared(np.asarray(train_dataset, dtype=config.floatX), borrow=True)
    train_y = shared(np.asarray(train_labels, dtype=config.floatX), borrow=True)
    test_x = shared(np.asarray(test_dataset, dtype=config.floatX), borrow=True)
    test_y = shared(np.asarray(test_labels, dtype=config.floatX), borrow=True)

    test_y = test_y.flatten()
    train_y = train_y.flatten()

    test_y = T.cast(test_y, 'int32')
    train_y = T.cast(train_y, 'int32')

    return train_x, train_y, test_x, test_y

