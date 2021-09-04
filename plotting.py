import numpy as np
import matplotlib.pyplot as plt


def plot(training_round):

    targeted_accuracy=np.load('./tp_100_benign_acc.npy')
    targeted_accuracy1=np.load('./tp_100_mal_acc.npy')
    sr = np.load('./tp_100_sr.npy')

    plt.plot(training_round,targeted_accuracy,'r',label='benign')
    plt.plot(training_round,targeted_accuracy1,'b',label='with adversary')
    plt.plot(training_round,sr,'g',label='success rate')
    plt.xlabel('training_round')
    plt.ylabel('accuracy')
    plt.legend(loc="upper right")
    plt.savefig('acccc.png')

    plt.show()

arr = np.arange(1,101)
plot(arr)