import matplotlib.pyplot as plt

from ReadDatabase import *
from lenet_5.Lenet5 import Lenet5

rng = np.random


'''os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #Recomendado quando profile=True, deixa mais lento porém é mais preciso
config.profile = True
config.profile_memory=False
config.profile_optimizer=True
config.debugprint = True'''


def main(batch_size=200, epocha=1, eta=0.1):
    database = read_files()

    #train_set_x, train_set_y = database[0]
    #test_set_x, test_set_y = database[1]

    lenet5 = Lenet5()
    lenet5.run_train(database, epochs=30)


def test():
    database = read_files()
    train_set_x, train_set_y = database[0]
    test_set_x, test_set_y = database[1]

    imgs = [train_set_x.get_value()[i].reshape(32, 32) for i in range(0, 5)]

    j = 0
    for img in imgs:
        plt.imshow(img)
        plt.show()
        print(train_set_y[j].eval())
        j += 1

if "__main__" == __name__:
    #params = sys.argv
    #create_binary_files()
    #x, y = get_files()
    main()
    #test()
