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


def finding_optimal_synthetic(num_users=100, kappa=10, dim = 40, noise_ratio=0.05):
    
    powers = - np.log(kappa) / np.log(dim) / 2
    DIM = np.arange(dim)
    S = np.power(DIM+1, powers)

    # Creat list data for all users 
    X_split = [[] for _ in range(num_users)]  # X for each user
    y_split = [[] for _ in range(num_users)]  # y for each user
    samples_per_user = np.random.lognormal(4, 2, num_users).astype(int) + 500
    indices_per_user = np.insert(samples_per_user.cumsum(), 0, 0, 0)
    num_total_samples = indices_per_user[-1]

    # Create mean of data for each user, each user will have different distribution
    mean_X = np.array([np.random.randn(dim) for _ in range(num_users)])

    # Covariance matrix for X
    X_total = np.zeros((num_total_samples, dim))
    y_total = np.zeros(num_total_samples)

    for n in range(num_users):
        # Generate data
        X_n = np.random.multivariate_normal(mean_X[n], np.diag(S), samples_per_user[n])
        X_total[indices_per_user[n]:indices_per_user[n+1], :] = X_n

    # Normalize all X's using LAMBDA
    norm = np.sqrt(np.linalg.norm(X_total.T.dot(X_total), 2) / num_total_samples)
    X_total /= norm

    # Generate weights and labels
    W = np.random.rand(dim)
    y_total = X_total.dot(W)
    noise_variance = 0.01
    y_total = y_total + np.sqrt(noise_ratio) * np.random.randn(num_total_samples)


    # finding optimal
    model = LinearRegression()
    model.fit(X_total, y_total)
    out = model.predict(X_total)
    LOSS = sk.metrics.mean_squared_error(out,y_total)
    return LOSS , model.coef_, model.intercept_

def main():
    loss = 0
    loss, w, b = finding_optimal_synthetic()
    print("loss for all data", loss)
    print("w",w)
    print("b",b)

if __name__ == "__main__":
    main()

