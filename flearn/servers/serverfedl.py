import torch
import os

from flearn.users.userfedl import UserFEDL
from flearn.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FEDL(Server):
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                 local_epochs, optimizer, num_users, times):
        super().__init__(dataset,algorithm, model[0], batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserFEDL(id, train, test, model, batch_size, learning_rate, hyper_learning_rate, L, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()
            self.send_grads()
            # Evaluate model each interation
            self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()
            self.aggregate_grads()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()