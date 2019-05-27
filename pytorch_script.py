import torch
from torch.utils import data

from my_classes import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True
path = './data1'

# Parameters
params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 50

