#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    accuracy={}
    #Baseline
    num_classes = 10
    batch_size = 32
    lr =0.1
    momentum=0
    LeakyReLU=0
    accuracy['Baseline']=oldmain(num_classes,batch_size,lr,momentum,LeakyReLU)

    #Size 64
    batch_size = 64
    lr =0.1
    momentum=0
    LeakyReLU=0
    accuracy['Size64']=oldmain(num_classes,batch_size,lr,momentum,LeakyReLU)

    #LR 0.01
    batch_size = 32
    lr =0.01
    momentum=0
    LeakyReLU=0
    accuracy['LR001']=oldmain(num_classes,batch_size,lr,momentum,LeakyReLU)

    #Momentum 0.9
    batch_size = 32
    lr =0.1
    momentum=0.9
    LeakyReLU=0
    accuracy['Momentump9']=oldmain(num_classes,batch_size,lr,momentum,LeakyReLU)
    #LeakyReLU
    batch_size = 32
    lr =0.1
    momentum=0
    LeakyReLU=1
    accuracy['LeakyReLU']=oldmain(num_classes,batch_size,lr,momentum,LeakyReLU)

    print(accuracy)

def oldmain(classes,batch,eta,momentum,LeakyReLU):
    # Load the dataset
    num_classes = classes
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = batch
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    if not LeakyReLU:
        model = nn.Sequential(
                  nn.Linear(784, 128),
                  nn.ReLU(),
                  nn.Linear(128, 10),
                )
    else:
        model = nn.Sequential(
                  nn.Linear(784, 128),
                  nn.LeakyReLU(),
                  nn.Linear(128, 10),
                )
    lr=eta
    #momentum=0
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
    return accuracy

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
