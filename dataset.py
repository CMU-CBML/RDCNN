import torch
from torch.utils import data
import os
import glob
import pandas as pd
import numpy as np
import re


def read_files_array(filename):
    # f = open(filename, 'r')
    # row_data = []
    # row_data = [line.split() for line in f]

    # matrix = np.array([])
    # matrix.resize((21, 21))

    # for i in range(len(row_data)):
    #     col = int(row_data[i][0])
    #     row = int(row_data[i][1])
    #     val = float(row_data[i][2])
    #     matrix[col][row] = val
    # f.close()

    row_data = np.loadtxt(filename)
    matrix = np.resize(row_data[:,2],(21,21))

    return matrix

# 4-channel Dataset
# class rdDataset(data.Dataset):
#   'Characterizes a dataset for PyTorch'

#   def __init__(self, path_data):
#         'Initialization'
#         self.path_data = path_data
#         self.filename_output_all = glob.glob(path_data + '/output/*.txt')
#         self.para = pd.read_csv(
#             path_data + '/dataset_DKtGeo.txt', sep="\t", header=None)
#         self.para = self.para.rename(
#             columns={0: 'file_num', 1: 'D', 2: 'K', 3: 'T', 8: 'Geo'})

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.filename_output_all)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         filename_output = self.filename_output_all[index]
#         file_num = int(re.search(r'\d+', filename_output).group(0))
#         # Load data and get label
#         t = self.para.loc[file_num]['T']
#         k = self.para.loc[file_num]['K']
#         d = self.para.loc[file_num]['D']

#         matrix_input = np.array([])
#         matrix_input.resize((4, 21, 21))
#         filename_input = self.path_data + "/input/geometry_" + \
#             str(int(self.para.loc[file_num]['Geo']))+"_input.txt"
#         matrix_input[0] = read_files_array(filename_input)
#         matrix_input[1] = t
#         matrix_input[2] = k
#         matrix_input[3] = d

#         matrix_output = read_files_array(filename_output)
#         matrix_output.resize(1, 21, 21)

#         return matrix_input, matrix_output

# Using file name to decide dataset size
# class rdDataset(data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, path_data):
#         'Initialization'
#         self.path_data = path_data
#         self.filename_output_all = glob.glob(path_data + '/output/*.txt')
#         self.para = pd.read_csv(path_data + '/dataset_DKtGeo.txt', sep="\t", header=None)
#         self.para=self.para.rename(columns = {0:'file_num',1:'D',2:'K',3:'T',8:'Geo'})
#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.filename_output_all)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         filename_output = self.filename_output_all[index]
#         file_num = int(re.search(r'\d+', filename_output).group(0))
#         # Load data and get label
#         t = self.para.loc[file_num]['T']
#         k = self.para.loc[file_num]['K']
#         d = self.para.loc[file_num]['D']
           
#         matrix_input = np.array([])
#         matrix_input.resize((5, 21, 21))
#         filename_input = self.path_data + "/input/geometry_"+str(int(self.para.loc[file_num]['Geo']))+"_input.txt"
#         matrix_input[0] = read_files_array(filename_input)
#         matrix_input[1] = t
#         matrix_input[2] = k
#         matrix_input[3] = d
#         if matrix_input[0].all()>-2:
#             matrix_input[4] = 0
#         else:
#             matrix_input[4] = 1

#         matrix_output = read_files_array(filename_output)
#         matrix_output.resize(1, 21, 21)

#         return matrix_input, matrix_output

class rdDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path_data):
        'Initialization'
        self.path_data = path_data
        self.filename_output_all = glob.glob(path_data + '/output/*.txt')
        self.para = pd.read_csv(path_data + '/dataset_DKtGeo.txt', sep="\t", header=None)
        self.para = self.para.rename(columns = {0:'file_num',1:'D',2:'K',3:'T',4:'U0',5:'U1',6:'U2',7:'U3',8:'Geo', 9:'ParaSet'})
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
        paraset = self.para.loc[index]['ParaSet']

        filename_output = self.path_data + "/output/mesh_"+str(int(file_num))+".txt"
           
        matrix_input = np.array([])
        matrix_input.resize((4, 21, 21))
        filename_input = self.path_data + "/input/geometry_"+str(int(geo))+ "_" +str(int(paraset)) + "_input.txt"
        matrix_input[0] = read_files_array(filename_input)
        matrix_input[1] = t
        matrix_input[2] = k
        matrix_input[3] = d

        matrix_output = read_files_array(filename_output)
        matrix_output.resize(1, 21, 21)

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