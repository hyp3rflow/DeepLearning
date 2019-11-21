import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

from example.mnist import *
from example.TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=x_train.shape[1]
    , hidden_size=50, output_size=t_train.shape[1])

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
 
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

plt.plot(range(train_acc_list.__len__()), train_acc_list, label="train")
plt.plot(range(test_acc_list.__len__()), test_acc_list, linestyle="--", label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title('train & test acc')
plt.legend()
plt.show()