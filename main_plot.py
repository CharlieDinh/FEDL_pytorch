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
    num_glob_iters = 200
    dataset = "Linear_synthetic"
    local_ep = [20,20,20,20]
    L = [15,15,15,15]
    learning_rate = [0.01, 0.01, 0.01]
    hyper_learning_rate =  [0.1, 0.5, 0.7]
    batch_size = [0,0,0]
    algorithms = ["FEDL","FEDL","FEDL"]
    plot_summary_one_figure(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
