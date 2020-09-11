#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.serveravg import FedAvg
from flearn.servers.serverfedl import FEDL
from flearn.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
         local_epochs, optimizer, clients_per_round, rho, times):

    for i in range(times):
        print("---------------Running time:------------",i)

        # Generate model
        if(model == "mclr"): #for Mnist and Femnist datasets
            model = Mclr_Logistic(), model

        if(model == "linear"): # For Linear dataset
            model = Linear_Regression(40,1), model
        
        if(model == "dnn"): # for Mnist and Femnist datasets
            model = model = DNN(), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters, local_epochs, optimizer, clients_per_round, rho, i)
        
        if(algorithm == "FEDL"):
            server = FEDL(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters, local_epochs, optimizer, clients_per_round, rho, i)
        server.train()
        server.test()

    # Average data 
    average_data(num_users=clients_per_round, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms=algorithm, batch_size=batch_size, dataset=dataset, rho = rho, times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Femnist", "Linear_synthetic", "Logistic_synthetic"])
    parser.add_argument("--model", type=str, default="mclr", choices=["linear", "mclr", "dnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Local learning rate")
    parser.add_argument("--hyper_learning_rate", type=float, default = 0, help=" Learning rate of FEDL")
    parser.add_argument("--L", type=int, default=0, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FEDL",choices=["FEDL", "FedAvg"])
    parser.add_argument("--clients_per_round", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--rho", type=float, default=0, help="Conditon Number")
    parser.add_argument("--times", type=int, default=1, help="running time")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Hyper learing rate       : {}".format(args.hyper_learning_rate))
    print("Subset of users      : {}".format(args.clients_per_round))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hyper_learning_rate = args.hyper_learning_rate, 
        L = args.L,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        clients_per_round = args.clients_per_round,
        rho = args.rho,
        times = args.times
        )
