import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from flearn.users.userbase import User

# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, hyper_learning_rate, lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, hyper_learning_rate, lamda,
                         local_epochs)

        if(model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]


    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            loss_per_epoch = 0
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                loss_per_epoch += loss 
            LOSS += loss_per_epoch
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS

    # def train(self, epochs):
    #     LOSS = 0
    #     self.model.train()
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()
    #         self.optimizer.step()
    #         self.clone_model_paramenter(self.model.parameters(), self.local_model)
    #     return LOSS

