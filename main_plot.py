#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

if(1): # plot for MNIST convex 
    numusers = 10
    num_glob_iters = 800
    dataset = "Mnist"
    local_ep = [20,20,20,20]
    L = [15,15,15,15]
    learning_rate = [0.005, 0.005]
    hyper_learning_rate =  [0.2, 0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.1,0.1]
    algorithms = ["FedAvg"]
    plot_summary_one_figure(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
