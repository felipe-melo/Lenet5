from notMNIST.notMNISTDataset import download_no_mnist, maybe_extract, maybe_pickle, plotingSomeImg, merge_datasets, np, \
    pickle, os, extract_overlap_hash_where, sanitize, save_pickle_file, plt, create_dataset, load_dataset
from lenet_5.Lenet5 import Lenet5
from util.Constants import Constants


def my_test():

    pickle_file = os.path.join(Constants.dataset_root, 'notMNIST.pickle')

    fileObj = open(pickle_file, 'rb')

    allDatabase = pickle.load(fileObj)

    train_dataset = allDatabase['train_dataset']
    train_labels = allDatabase['train_labels']
    valid_dataset = allDatabase['valid_dataset']
    valid_labels = allDatabase['valid_labels']
    test_dataset = allDatabase['test_dataset']
    test_labels = allDatabase['test_labels']

    print('Training:', train_dataset.shape, 'Label:', train_labels.shape)
    print('Validation:', valid_dataset.shape, 'Label:', valid_labels.shape)
    print('Testing:', test_dataset.shape, 'Label:', test_labels.shape)

    for i in range(10):
        sample_idx = np.random.randint(train_dataset.shape[0])  # pick a random image index
        sample_image = train_dataset[sample_idx]  # extract a 2D slice
        print(train_labels[sample_idx])
        plt.figure()
        plt.imshow(sample_image.reshape(32, 32))
        plt.show()

    for i in range(10):
        sample_idx = np.random.randint(valid_dataset.shape[0])  # pick a random image index
        sample_image = valid_dataset[sample_idx]  # extract a 2D slice
        print(valid_labels[sample_idx])
        plt.figure()
        plt.imshow(sample_image.reshape(32, 32))
        plt.show()

    for i in range(10):
        sample_idx = np.random.randint(test_dataset.shape[0])  # pick a random image index
        sample_image = test_dataset[sample_idx]  # extract a 2D slice
        print(test_labels[sample_idx])
        plt.figure()
        plt.imshow(sample_image.reshape(32, 32))
        plt.show()


def main():
    train_set_x, train_set_y, valid_x, valid_y, test_set_x, test_set_y = load_dataset()

    lenet5 = Lenet5()
    lenet5.run_train(train_set_x, train_set_y, test_set_x, test_set_y, valid_dataset=valid_x, valid_labels=valid_y, epochs=5)


if "__main__" == __name__:
    main()
    #create_dataset()
    #my_test()
