from experiment.MergeDatasets import *
from lenet_5.Lenet5 import Lenet5

rng = np.random


def main():
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

    '''database = read_files()
    train_set_x, train_set_y = database[0]
    test_set_x, test_set_y = database[1]'''

    lenet5 = Lenet5(labels=20)
    lenet5.run_train(train_set_x, train_set_y, test_set_x, test_set_y, epochs=5)


if "__main__" == __name__:
    main()
    #create_binary_files()
    #plotingSomeImg()