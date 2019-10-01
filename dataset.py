import torch
from torch.utils import data
import os
import glob
import pandas as pd
import numpy as np
import h5py
import re
import sys


def read_files_array(filename):
#     row_data = np.loadtxt(filename)
#     matrix = np.resize(row_data[:,2],(21,21))

    row_data = pd.read_csv(filename,sep=' ', header=None)
    row_data = row_data.iloc[0:,2].values    
    matrix = row_data.astype('float').reshape(21,21)
    return matrix

class H5Dataset_new(data.Dataset):

    def __init__(self, path):
        super(H5Dataset_new, self).__init__()
        self.file_path =path
        self.data = None
        self.target = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["input"])

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.file_path, 'r')["input"]
        if self.target is None:
            self.target = h5py.File(self.file_path, 'r')["output"]         
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.dataset_len

class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path, 'r', swmr = True)
        self.data = h5_file.get('input')
        self.target = h5_file.get('output')

    def __getitem__(self, index):            
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.data.shape[0]

class rdDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path_data):
        'Initialization'
        self.path_data = path_data
        self.filename_output_all = glob.glob(path_data + '/output/*.txt')
        self.para = pd.read_csv(path_data + '/dataset_DKtGeo.txt', sep="\t", header=None).values
      #   self.para = self.para.rename(columns = {0:'file_num',1:'D',2:'K',3:'T',4:'U0',5:'U1',6:'U2',7:'U3',8:'Geo', 9:'ParaSet'})
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.para)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        # Load data and get label
        file_num = self.para[index,0]
        t = self.para[index,3]
        k = self.para[index,2]
        d = self.para[index,1]
      #   u0 = self.para.loc[index]['U0']
      #   u1 = self.para.loc[index]['U1']
      #   u2 = self.para.loc[index]['U2']
      #   u3 = self.para.loc[index]['U3']
        geo = self.para[index,8]
        paraset = self.para[index,9]

        filename_output = self.path_data + "/output/mesh_"+str(int(file_num))+".txt"
           
        matrix_input = torch.zeros([4,21,21], dtype = torch.float)
        filename_input = self.path_data + "/input/geometry_"+str(int(geo))+ "_" +str(int(paraset)) + "_input.txt"
        matrix_input[0] = torch.from_numpy(read_files_array(filename_input))
        matrix_input[1] = t
        matrix_input[2] = k
        matrix_input[3] = d

        matrix_output = torch.zeros([1,21,21], dtype = torch.float)
        matrix_output = torch.from_numpy(read_files_array(filename_output))
      #   matrix_output.resize(1, 21, 21)

        return matrix_input, matrix_output

class rdDataset_old(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path_data):
        'Initialization'
        self.path_data = path_data
        self.filename_output_all = glob.glob(path_data + '/output/*.txt')
        self.para = pd.read_csv(path_data + '/dataset_DKtGeo.txt', sep="\t", header=None)
        self.para = self.para.rename(columns = {0:'file_num',1:'D',2:'K',3:'T',4:'U0',5:'U1',6:'U2',7:'U3',8:'Geo'})
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.para)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        # Load data and get label
        file_num = self.para.loc[index]['file_num']
        t = self.para.loc[index]['T']
        k = self.para.loc[index]['K']
        d = self.para.loc[index]['D']
      #   u0 = self.para.loc[index]['U0']
      #   u1 = self.para.loc[index]['U1']
      #   u2 = self.para.loc[index]['U2']
      #   u3 = self.para.loc[index]['U3']
        geo = self.para.loc[index]['Geo']
      #   paraset = self.para.loc[index]['ParaSet']

        filename_output = self.path_data + "/output/mesh_"+str(int(file_num))+".txt"
           
        matrix_input = np.array([])
        matrix_input.resize((4, 21, 21))
        filename_input = self.path_data + "/input/geometry_"+str(int(geo))+ "_input.txt"
        matrix_input[0] = read_files_array(filename_input)
        matrix_input[1] = t
        matrix_input[2] = k
        matrix_input[3] = d

        matrix_output = read_files_array(filename_output)
        matrix_output.resize(1, 21, 21)

        return matrix_input, matrix_output