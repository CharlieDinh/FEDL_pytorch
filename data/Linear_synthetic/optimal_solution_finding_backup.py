import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn as sk
np.random.seed(0)

NUM_USER = 100

def normalize_data(X):

    #nomarlize all feature of data between (-1 and 1)
    normX = X - X.min()
    normX = normX / (X.max() - X.min())

    # nomarlize data with respect to -1 < X.X^T < 1.
    temp = normX.dot(normX.T)
    return normX/np.sqrt(temp.max())


def finding_optimal_synthetic(alpha = 0.5, hyper_learning_rate = 0.5, iid = 0):

    # Generate parameters for controlling kappa 
    dimension = 60
    NUM_CLASS = 1
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 100
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, hyper_learning_rate, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.random.normal(B[i], 1, dimension)

    data_all_x = []
    data_all_y = []

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        nom_xx = normalize_data(xx)
        
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            yy[j] = np.dot(nom_xx[j], W) + b

        data_all_x.extend(nom_xx)
        data_all_y.extend(yy)


    # finding optimal
    model = LinearRegression()
    model.fit(data_all_x, data_all_y)
    out = model.predict(data_all_x)
    LOSS = sk.metrics.mean_squared_error(out,data_all_y)
    return LOSS , model.coef_, model.intercept_

def main():
    loss = 0
    loss, w, b = finding_optimal_synthetic(alpha=0.5, hyper_learning_rate=0.5)
    print("loss for all data", loss)
    print("w",w)
    print("b",b)

if __name__ == "__main__":
    main()
