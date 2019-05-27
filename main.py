import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from matplotlib import pyplot as plt

import data
from dataset import rdDataset
from model import rdcnn_2

# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--path', type=os.path.abspath, required=True, help="data path")
# parser.add_argument('--test_ratio', type=float, required=True, help="percentage of test data")
# parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
# parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default= 100, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
# parser.add_argument('--dropout_rate', type=float, default=0.0, help="dropout ratio")
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
# parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
# opt = parser.parse_args()
#
# # CUDA for PyTorch
# if opt.cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")
#
# torch.manual_seed(opt.seed)
#
# device = torch.device("cuda" if opt.cuda else "cpu")
#
#
#
# # cudnn.benchmark = True
# path = opt.path
# print(path)
#
# # Parameters
#
# params = {'test_split': opt.test_ratio,
#           'shuffle_dataset': True,
#           'batchsize': opt.batchSize,
#           'testBatchsize':opt.testBatchSize,
#           'random_seed': opt.seed,
#           'numworkers':opt.threads,
#           'pinmemory':True}
# max_epoches = opt.nEpochs
# learning_rate = opt.lr
# drop_rate = opt.dropout_rate

# For debugging
import sys
sys.path.append("pydevd-pycharm.egg")

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=8081, stdoutToServer=True, stderrToServer=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
path = './data_bc'
params = {'test_split': .25,
          'shuffle_dataset': True,
          'batchsize': 32,
          'testBatchsize': 10,
          'random_seed': 42,
          'numworkers':32,
          'pinmemory':True}
max_epoches = 100
learning_rate = 1e-3
drop_rate = 0.0

print('===> Loading datasets')
# Load All Dataset
dataset = rdDataset(path)

# Creating data indices for training and validation splits:
training_data_loader, testing_data_loader = data.DatasetSplit(dataset, **params)

print('===> Building model')
# model = rdcnn(drop_rate).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model = rdcnn_2(drop_rate).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    #         print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch, epoch_loss / len(training_data_loader)


def test():
    avg_error = 0
    avg_loss = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)

            prediction = model(input)
            tmp_error = 0
            #             print(len(prediction))
            for j in range(len(prediction)):
                tmp_error += torch.mean((prediction[j] - target[j]) ** 2 / torch.max(target[j]))
            avg_error += tmp_error / len(prediction)
            mse = criterion(prediction, target)
            avg_loss += mse
    print("===> Avg. Loss: {:.4f} ".format(avg_loss / len(testing_data_loader)))
    print("===> Avg. Error: {:.4f} ".format(avg_error / len(testing_data_loader)))
    return avg_loss / len(testing_data_loader), avg_error / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = "./checkpoint_databc1/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# %%
L_train_loss = []
L_test_loss = []
L_test_error = []
for epoch in range(1, max_epoches + 1):
    train_loss = train(epoch)
    test_loss, test_error = test()
    checkpoint(epoch)
    #     data.TestErrorPlot(model,device, testing_data_loader)
    L_train_loss.append(train_loss)
    L_test_loss.append(test_loss)
    L_test_error.append(test_error)

# def train(epoch):
#     epoch_loss = 0
#     for iteration, batch in enumerate(training_data_loader, 1):
#         input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)
#         optimizer.zero_grad()
#         loss = criterion(model(input), target)
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#
# #         print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
#
#     print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
#     return epoch, epoch_loss / len(training_data_loader)
#
# def test():
#     avg_error = 0
#     avg_loss = 0
#     with torch.no_grad():
#         for batch in testing_data_loader:
#             input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)
#
#             prediction = model(input)
#             tmp_error = 0
# #             print(len(prediction))
#             for j in range(len(prediction)):
#                 tmp_error += torch.mean((prediction[j]-target[j])**2/torch.max(target[j]))
#             avg_error += tmp_error / len(prediction)
#             mse = criterion(prediction, target)
#             avg_loss += mse
#     print("===> Avg. Loss: {:.4f} ".format(avg_loss / len(testing_data_loader)))
#     print("===> Avg. Error: {:.4f} ".format(avg_error / len(testing_data_loader)))
#     return avg_loss / len(testing_data_loader),avg_error / len(testing_data_loader)
#
# def checkpoint(epoch):
#     model_out_path = "./checkpoint_largedata2/model_epoch_{}.pth".format(epoch)
#     torch.save(model, model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
#
# L_train_loss = []
# L_test_loss = []
# L_test_error = []
# for epoch in range(1, max_epoches + 1):
#     train_loss = train(epoch)
#     test_loss,test_error = test()
#     checkpoint(epoch)
#     data.TestErrorPlot(model,device, testing_data_loader)
#     L_train_loss.append(train_loss)
#     L_test_loss.append(test_loss)
#     L_test_error.append(test_error)

# data.TestErrorPlot(model,device, testing_data_loader)


# with torch.no_grad():
#     for batch in testing_data_loader:
#         input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)

#         prediction = model(input)
        
# for t in range(len(prediction)):
#     fig, ax = plt.subplots(1,2, figsize=(10,5))

#     im = ax[0].imshow(prediction[t][0].cpu(),cmap = "jet")
#     im = ax[1].imshow(target[t][0].cpu(),cmap = "jet")

#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.84, 0.27, 0.01, 0.47])
#     fig.colorbar(im, cax=cbar_ax)

# plt.show()