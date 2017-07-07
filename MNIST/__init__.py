from MNIST.MNISTDataset import *
from lenet_5.Lenet5 import Lenet5

rng = np.random


'''os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #Recomendado quando profile=True, deixa mais lento porém é mais preciso
config.profile = True
config.profile_memory=False
config.profile_optimizer=True
config.debugprint = True'''


def main():
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

    '''database = read_files()
    train_set_x, train_set_y = database[0]
    test_set_x, test_set_y = database[1]'''

    lenet5 = Lenet5()
    lenet5.run_train(train_set_x, train_set_y, test_set_x, test_set_y, epochs=30)


if "__main__" == __name__:
    main()
    #create_binary_files()
    #plotingSomeImg()
