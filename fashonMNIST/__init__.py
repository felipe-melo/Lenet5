from fashonMNIST.MNISTDataset import *
from lenet_5.Lenet5 import Lenet5
import sys, time

rng = np.random


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #Recomendado quando profile=True, deixa mais lento porém é mais preciso
#config.profile = True
#config.profiling.ignore_first_call = True

t0 = time.time()

def main(_epochs):
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

    '''database = read_files()
    train_set_x, train_set_y = database[0]
    test_set_x, test_set_y = database[1]'''

    lenet5 = Lenet5()
    lenet5.run_train(train_set_x, train_set_y, test_set_x, test_set_y, t0=t0, epochs=_epochs)


if "__main__" == __name__:
    epochs = int(sys.argv[1])

    main(epochs)
    #create_binary_files()
    #plotingSomeImg()
