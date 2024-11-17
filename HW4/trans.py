import numpy as np
import random
import matplotlib.pyplot as plt
import urllib.request
import os

from tqdm import tqdm
from sklearn.datasets import load_svmlight_file

url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale"
path = "./data/cpusmall_scale"

def squared_error(X, y, w):
    predictions = X @ w
    return np.mean((predictions - y) ** 2)

def polynomial_transform(X, Q=3):
    X_poly = X
    for q in range(2, Q + 1):
        X_poly = np.hstack((X_poly, X[:, 1:] ** q))
    return X_poly

if __name__ == "__main__":
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    X, y = load_svmlight_file(path)
    X = X.toarray()
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    N = 64
    num_experiments = 1126
    eta = 0.01
    iterations = 100000

    E_in_lin_all, E_out_lin_all = [], []
    E_in_poly_all, E_out_poly_all = [], []
    E_in_sgd_all, E_out_sgd_all = [], []

    for experiment in tqdm(range(num_experiments), desc="Experiments"):
        indices = np.random.choice(X.shape[0], N, replace=False)
        X_train, y_train = X[indices], y[indices]
        X_test, y_test = np.delete(X, indices, axis=0), np.delete(y, indices)

        w_lin = np.linalg.pinv(X_train) @ y_train
        E_in_lin = squared_error(X_train, y_train, w_lin)
        E_out_lin = squared_error(X_test, y_test, w_lin)
        E_in_lin_all.append(E_in_lin)
        E_out_lin_all.append(E_out_lin)

        X_poly_train = polynomial_transform(X_train)
        X_poly_test = polynomial_transform(X_test)
        w_poly = np.linalg.pinv(X_poly_train) @ y_train
        E_in_poly = squared_error(X_poly_train, y_train, w_poly)
        E_out_poly = squared_error(X_poly_test, y_test, w_poly)
        E_in_poly_all.append(E_in_poly)
        E_out_poly_all.append(E_out_poly)

        w_sgd = np.zeros(X.shape[1])
        E_in_sgd, E_out_sgd = [], []

        for t in range(1, iterations + 1):
            i = random.randint(0, N - 1)
            xi, yi = X_train[i], y_train[i]

            gradient = 2 * (xi @ w_sgd - yi) * xi
            w_sgd -= eta * gradient

            if t % 200 == 0:
                E_in_sgd.append(squared_error(X_train, y_train, w_sgd))
                E_out_sgd.append(squared_error(X_test, y_test, w_sgd))

        E_in_sgd_all.append(E_in_sgd)
        E_out_sgd_all.append(E_out_sgd)

    avg_E_in_lin = np.mean(E_in_lin_all)
    avg_E_out_lin = np.mean(E_out_lin_all)
    avg_E_in_poly = np.mean(E_in_poly_all)
    avg_E_out_poly = np.mean(E_out_poly_all)

    avg_E_in_sgd = np.mean(E_in_sgd_all, axis=0)
    avg_E_out_sgd = np.mean(E_out_sgd_all, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(200, iterations + 1, 200), avg_E_in_sgd, label='$E_{in}(w_t)$', color='red')
    plt.plot(range(200, iterations + 1, 200), avg_E_out_sgd, label='$E_{out}(w_t)$', color='blue')
    plt.axhline(y=avg_E_in_lin, color='orange', label='$E_{in}(w_{lin})$')
    plt.axhline(y=avg_E_out_lin, color='green', label='$E_{out}(w_{lin})$')
    plt.xlabel('Iterations (t)')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error vs Iterations for SGD and Linear Regression')

    Ein_diff = np.array(E_in_lin_all) - np.array(E_in_poly_all)
    avg_Ein_diff = np.mean(Ein_diff)
    plt.figure(figsize=(10, 6))
    plt.hist(Ein_diff, bins=30, color='orange')
    plt.xlabel('$E_{in}^{sqr}(w_{lin}) - E_{in}^{sqr}(w_{poly})$')
    plt.ylabel('Frequency')
    plt.title('Histogram of $E_{{in}}^{{sqr}}(w_{{lin}}) - E_{{in}}^{{sqr}}(w_{{poly}})$\nAverage: {:.4f}'.format(avg_Ein_diff))

    Eout_diff = np.array(E_out_lin_all) - np.array(E_out_poly_all)
    avg_Eout_diff = np.mean(Eout_diff)
    plt.figure(figsize=(10, 6))
    plt.hist(Eout_diff, bins=30, color='green')
    plt.xlabel('$E_{out}^{sqr}(w_{lin}) - E_{out}^{sqr}(w_{poly})$')
    plt.ylabel('Frequency')
    plt.title('Histogram of $E_{{out}}^{{sqr}}(w_{{lin}}) - E_{{out}}^{{sqr}}(w_{{poly}})$\nAverage: {:.4f}'.format(avg_Eout_diff))
    plt.show()
