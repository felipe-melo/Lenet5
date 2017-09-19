import hashlib
import os
from scipy import ndimage
import tarfile
from theano import config, shared
import theano.tensor as T
import sys
from util.Constants import Constants
import numpy as np
from urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt


url = 'https://commondatastorage.googleapis.com/books1000/'


def download_no_mnist(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(Constants.dataset_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(Constants.dataset_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            img = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            image_data = (np.zeros((32, 32)).astype(float) - pixel_depth / 2) / pixel_depth

            for i in range(28):
                for j in range(28):
                    image_data[i+2][j+2] = img[i][j]

            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def plotingSomeImg(train_datasets):

    for train in train_datasets:
        pickle_file = train
        with open(pickle_file, 'rb') as f:
            letter_set = pickle.load(f)  # unpickle
            sample_idx = np.random.randint(len(letter_set))  # pick a random image index
            sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
            plt.figure()
            plt.imshow(sample_image)  # display it
            plt.show()


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def extract_overlap_hash_where(dataset_1, dataset_2):
    dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
    dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
    overlap = {}
    for i, hash1 in enumerate(dataset_hash_1):
        duplicates = np.where(dataset_hash_2 == hash1)
        if len(duplicates[0]):
            overlap[i] = duplicates[0]
    return overlap


def sanitize(dataset_1, dataset_2, labels_1):
    dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
    dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
    overlap = []  # list of indexes
    for i, hash1 in enumerate(dataset_hash_1):
        duplicates = np.where(dataset_hash_2 == hash1)
        if len(duplicates[0]):
            overlap.append(i)
    return np.delete(dataset_1, overlap, 0), np.delete(labels_1, overlap, None)


def save_pickle_file(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):

    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    pickle_file = os.path.join(Constants.dataset_root, 'notMNIST.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset.reshape(train_dataset.shape[0], 1024),
            'train_labels': train_labels,
            'valid_dataset': valid_dataset.reshape(valid_dataset.shape[0], 1024),
            'valid_labels': valid_labels,
            'test_dataset': test_dataset.reshape(test_dataset.shape[0], 1024),
            'test_labels': test_labels
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def create_dataset():
    train_filename = download_no_mnist('notMNIST_large.tar.gz', 247336696)
    test_filename = download_no_mnist('notMNIST_small.tar.gz', 8458043)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

    train_size = 350000
    valid_size = 15000
    test_size = 15000

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    '''overlap_test_train = extract_overlap_hash_where(test_dataset, train_dataset)
    print('Number of overlaps:', len(overlap_test_train.keys()))

    overlap_valid_train = extract_overlap_hash_where(valid_dataset, train_dataset)
    print('Number of overlaps:', len(overlap_valid_train.keys()))

    overlap_valid_test = extract_overlap_hash_where(valid_dataset, test_dataset)
    print('Number of overlaps:', len(overlap_valid_test.keys()))'''

    train_dataset, train_labels = sanitize(train_dataset, test_dataset, train_labels)
    train_dataset, train_labels = sanitize(train_dataset, valid_dataset, train_labels)
    test_dataset, test_labels = sanitize(test_dataset, valid_dataset, test_labels)

    save_pickle_file(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

    '''pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    allDatabase = pickle.load(open(pickle_file, 'rb'))

    train_dataset = allDatabase['train_dataset']
    valid_dataset = allDatabase['valid_dataset']
    test_dataset = allDatabase['test_dataset']'''

    '''overlap_test_train = extract_overlap_hash_where(test_dataset, train_dataset)
    print('Number of overlaps:', len(overlap_test_train.keys()))

    overlap_valid_train = extract_overlap_hash_where(valid_dataset, train_dataset)
    print('Number of overlaps:', len(overlap_valid_train.keys()))

    overlap_valid_test = extract_overlap_hash_where(valid_dataset, test_dataset)
    print('Number of overlaps:', len(overlap_valid_test.keys()))'''


def load_dataset():
    pickle_file = os.path.join(Constants.dataset_root, 'notMNIST.pickle')

    fileObj = open(pickle_file, 'rb')

    allDatabase = pickle.load(fileObj)

    train_dataset = allDatabase['train_dataset']
    train_labels = allDatabase['train_labels']
    valid_dataset = allDatabase['valid_dataset']
    valid_labels = allDatabase['valid_labels']
    test_dataset = allDatabase['test_dataset']
    test_labels = allDatabase['test_labels']

    fileObj.close()

    train_x = shared(np.asarray(train_dataset, dtype=config.floatX), borrow=True)
    train_y = shared(np.asarray(train_labels, dtype=config.floatX), borrow=True)
    valid_x = shared(np.asarray(valid_dataset, dtype=config.floatX), borrow=True)
    valid_y = shared(np.asarray(valid_labels, dtype=config.floatX), borrow=True)
    test_x = shared(np.asarray(test_dataset, dtype=config.floatX), borrow=True)
    test_y = shared(np.asarray(test_labels, dtype=config.floatX), borrow=True)

    test_y = test_y.flatten()
    train_y = train_y.flatten()
    valid_y = train_y.flatten()

    test_y = T.cast(test_y, 'int32')
    train_y = T.cast(train_y, 'int32')
    valid_y = T.cast(valid_y, 'int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y