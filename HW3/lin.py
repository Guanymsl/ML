import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import requests

url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale'
path = './data/cpusmall_scale'

def download(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)

def load(path):
    X, y = load_svmlight_file(path)
    X = X.toarray()
    return X, y

def hat(X):
    return np.linalg.pinv(X)

def mse(y1, y2):
    return np.mean((y1 - y2) ** 2)

def linreg(X, y, N, iterations):
    Ein_list = []
    Eout_list = []

    for _ in tqdm(range(iterations), desc="Iterations"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N)
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        w = hat(X_train) @ y_train

        y_train_pred = X_train @ w
        Ein = mse(y_train, y_train_pred)
        Ein_list.append(Ein)

        y_test_pred = X_test @ w
        Eout = mse(y_test, y_test_pred)
        Eout_list.append(Eout)

    return Ein_list, Eout_list

def linreg2(X, y, N, iterations):
    Ein_list = []
    Eout_list = []

    for _ in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N)
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        w = hat(X_train) @ y_train

        y_train_pred = X_train @ w
        Ein = mse(y_train, y_train_pred)
        Ein_list.append(Ein)

        y_test_pred = X_test @ w
        Eout = mse(y_test, y_test_pred)
        Eout_list.append(Eout)

    return Ein_list, Eout_list

if __name__ == "__main__":
    download(url, path)
    X, y = load(path)

    N = 32
    iterations = 1126

    print("Running Problem 10...")
    Ein_list, Eout_list = linreg(X, y, N, iterations)

    plt.figure()
    plt.title("Eout vs Ein")
    plt.scatter(Ein_list, Eout_list, color="purple", alpha=0.7)
    plt.xlabel('Ein')
    plt.ylabel('Eout')

    N_list = np.arange(25, 2001, 25)
    Ein_avg = []
    Eout_avg = []
    iterations = 16

    print("Running Problem 11...")
    for N in tqdm(N_list, desc="Iterations"):
        Ein_temp, Eout_temp = linreg2(X, y, N, iterations)
        Ein_avg.append(np.mean(Ein_temp))
        Eout_avg.append(np.mean(Eout_temp))

    plt.figure()
    plt.title("Ein and Eout vs N for d = 12")
    plt.plot(N_list, Ein_avg, label='Ein', color="blue")
    plt.plot(N_list, Eout_avg, label='Eout', color="red")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()

    X_two_features = X[:, :2]
    Ein_avg_2 = []
    Eout_avg_2 = []

    print("Running Problem 12...")
    for N in tqdm(N_list, desc="Iterations"):
        Ein_temp, Eout_temp = linreg2(X_two_features, y, N, iterations)
        Ein_avg_2.append(np.mean(Ein_temp))
        Eout_avg_2.append(np.mean(Eout_temp))

    plt.figure()
    plt.title("Ein and Eout vs N for d = 2")
    plt.plot(N_list, Ein_avg_2, label='Ein', color="orange")
    plt.plot(N_list, Eout_avg_2, label='Eout', color="green")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()

    plt.figure()
    plt.title("Ein and Eout vs N")
    plt.plot(N_list, Ein_avg, label='Ein1', color="blue")
    plt.plot(N_list, Eout_avg, label='Eout1', color="red")
    plt.plot(N_list, Ein_avg_2, label='Ein2', color="orange")
    plt.plot(N_list, Eout_avg_2, label='Eout2', color="green")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
