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

      