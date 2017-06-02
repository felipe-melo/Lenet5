#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import struct
import numpy as np
from theano import config, shared
import theano.tensor as T
import ast


train_images_file = "database/train-images.idx3-ubyte"
train_labels_file = "database/train-labels.idx1-ubyte"
test_images_file = "database/t10k-images.idx3-ubyte"
test_labels_file = "database/t10k-labels.idx1-ubyte"


def read_files(train_images=train_images_file, train_labels=train_labels_file, test_images=test_images_file,
               test_labels=test_labels_file):
    _train_x, _train_y = __read_file__(train_images, train_labels)
    _test_x, _test_y = __read_file__(test_images, test_labels)

    train_x = shared(np.asarray(_train_x, dtype=config.floatX), borrow=True)
    train_y = shared(np.asarray(_train_y, dtype=config.floatX), borrow=True)
    test_x = shared(np.asarray(_test_x, dtype=config.floatX), borrow=True)
    test_y = shared(np.asarray(_test_y, dtype=config.floatX), borrow=True)

    test_y = T.cast(test_y, 'int32')
    train_y = T.cast(train_y, 'int32')

    return [(train_x, train_y), (test_x, test_y)]


def __read_file__(image_x, label_y):
    img_file = open(image_x, 'rb').read()
    lab_file = open(label_y, 'rb').read()

    image_index = 0
    label_index = 0
    magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', img_file, image_index)
    magic, num_labels = struct.unpack_from('>II', lab_file, label_index)

    print('train_set:', num_images, ' label_set:', num_labels)

    data_x = [None] * num_images
    data_y = [None] * num_labels

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
    _train_x, _train_y = __read_file__(train_images_file, train_labels_file)
    _test_x, _test_y = __read_file__(test_images_file, test_labels_file)

    img_file_train = open('inputs/train_x', 'w')
    lab_file_train = open('inputs/train_y', 'w')

    img_file_test = open('inputs/test_x', 'w')
    lab_file_test = open('inputs/test_y', 'w')

    for xt, yt, x, y in zip(_test_x, _test_y, _train_x, _train_y):
        img_file_train.write(str(x.tolist()) + '\n')
        lab_file_train.write(np.array_str(y) + '\n')

        img_file_test.write(str(xt.tolist()) + '\n')
        lab_file_test.write(np.array_str(yt) + '\n')

    img_file_train.close()
    lab_file_train.close()
    img_file_test.close()
    lab_file_test.close()


def get_files():
    img_file = open('inputs/test_x', 'r').read()
    lab_file = open('inputs/test_y', 'r').read()

    data_x = [None]
    data_y = [None]

    index = 0

    for x, y in zip(img_file, lab_file):
        data_x[index] = np.asarray(ast.literal_eval(x), dtype=config.floatX)
        data_y[index] = np.asarray(y, dtype=config.floatX)

        index += 1
        break

    return data_x, data_y
