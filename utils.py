import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import pickle
import os
from numpy.linalg import norm
import torch
import math
from mxnet import nd
import mxnet as mx
import seaborn as sns




#seed = 10417617 # do not change or remove

def binary_data(inp):
    return (inp > 0.5) * 1.


def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1/(1+np.exp(-x))
    else:
        return 1/(1+torch.exp(-x))   

def shuffle_corpus(data):
    random_idx = np.random.permutation(len(data))
    return data[random_idx]



def generate_W(method, n_hidden, n_visible, p =0.1):
    if method == 'random':
    	W = torch.tensor(np.random.normal(0, np.sqrt(
            6.0 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)

    if method == 'real':
    	with open('RBM_k1.pickle', "rb") as handle:
            model = pickle.load(handle)
            W = model.W

    if method == 'sparse': 
        mulnoise = torch.tensor(np.random.normal(0, np.sqrt(
            6.0 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
        W = torch.tensor(mulnoise * np.random.binomial(1, p, (n_hidden, n_visible)), dtype=torch.float32)        

    return W


def log_sum_exp(x):
    c = nd.max(x).asscalar()
    return math.log(nd.sum(nd.exp(x - c)).asscalar()) + c

# https://stackoverflow.com/a/47521145
def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain
    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret



# def check_match(W, M):

#     Wo_dot_M = W.T @ M
#     latentsord = []
#     neuronord = []
#     # sort the neuron-latents pair by their cosine value (decreasing ordering)
#     sorted_ind = np.array(np.unravel_index(np.argsort(-Wo_dot_M, axis=None), Wo_dot_M.shape))

#     j = 0
#     while j < d: 
#         neuronnow = sorted_ind[0,0]
#         latentnow = sorted_ind[1,0]
#         latentsord.append(latentnow) 
#         neuronord.append(neuronnow)
#         sorted_ind = sorted_ind[:,sorted_ind[0,:]!=neuronnow]
#         sorted_ind = sorted_ind[:,sorted_ind[1,:]!=latentnow]
#         j+=1
#     z_est = np.sign(pred_rep) # (m, batch_size) m>=d, z:(d,batch_size)
#     match = []
#     mismatch = []
#     sparse = []
#     for i in range(z_est.shape[1]):
#         if np.sum(abs(z[:,i]))>0:
#             match.append(np.sum(abs(z[latentsord,i]*z_est[neuronord,i]))/np.sum(abs(z[:,i]))) # get the match between m and d
#             mismatch.append(np.sum((z[latentsord,i]==0)*abs(z_est[neuronord,i]))/max(np.sum(abs(z_est[:,i])),1))
#             sparse.append(np.mean(abs(z_est[:,i]))-np.mean(abs(z[:,i]))) 
#         else:
#             match.append(0)
#             mismatch.append(0)
#             sparse.append(0)
                
#     return np.mean(match), np.mean(mismatch), np.mean(sparse)    

      