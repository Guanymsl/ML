import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def load():
    data = []
    labels = []

    with open('./data/rcv1_train.binary', 'rb') as f:
        for i, line in enumerate(f):
            if i >= 200:
                break

            parts = line.split()
            label = int(parts[0])
            features = {int(k): float(v) for k, v in (x.decode('utf-8').split(":") for x in parts[1:])}
            features[0] = 1.0

            data.append(features)
            labels.append(label)

    return data, labels

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def pla(data, labels):
    dim = 47206
    w = np.zeros(dim)
    N = len(data)
    norms = []
    correct = 0
    updates = 0

    while correct < 5 * N:
        i = rd.randint(0, N - 1)
        x = np.zeros(dim)
        for ind, val in data[i].items():
            x[ind] = val
        y = labels[i]

        if sign(np.dot(w, x)) != y:
            w += y * x
            norms.append(np.linalg.norm(w))
            updates += 1
            correct = 0
        else:
            correct += 1

    return norms, updates

def modified_pla(data, labels):
    dim = 47206
    w = np.zeros(dim)
    N = len(data)
    correct = 0
    updates = 0

    while correct < 5 * N:
        i = rd.randint(0, N - 1)
        x = np.zeros(dim)
        for ind, val in data[i].items():
            x[ind] = val
        y = labels[i]

        if sign(np.dot(w, x)) == y:
            correct += 1
        else:
            while sign(np.dot(w, x)) != y:
                w += y * x
                updates += 1
            correct = 0

    return updates

if __name__ == "__main__":
    updates_list = []
    updates_list_2 = []
    norms_list = []
    Tmin = np.inf
    iterations = 1000
    data, labels = load()

    print("Running Original PLA...")
    for i in tqdm(range(iterations), desc="Iterations"):
        rd.seed(int(time.time()))
        l, cnt = pla(data, labels)
        updates_list.append(cnt)
        norms_list.append(l)
        Tmin = min(Tmin, cnt)

    print("Running Modified PLA...")
    for i in tqdm(range(iterations), desc="Iterations"):
        rd.seed(int(time.time()))
        cnt = modified_pla(data, labels)
        updates_list_2.append(cnt)

    plt.figure()
    plt.hist(updates_list, bins=30, color='blue')
    plt.title(f"Median of Original PLA: {np.median(updates_list):.1f}")
    plt.xlabel("Number of Updates")
    plt.ylabel("Frequency")

    plt.figure()
    plt.hist(updates_list_2, bins=30, color='orange')
    plt.title(f"Median of Modified PLA: {np.median(updates_list_2):.1f}")
    plt.xlabel("Number of Updates")
    plt.ylabel("Frequency")

    plt.figure()
    plt.hist(updates_list, bins=30, alpha=0.5, label="Original PLA", color='blue')
    plt.hist(updates_list_2, bins=30, alpha=0.5, label="Modified PLA", color='orange')
    plt.xlabel("Number of Updates")
    plt.ylabel("Frequency")
    plt.legend()

    plt.figure()
    for i in range(iterations):
        plt.plot(range(1, Tmin + 1), norms_list[i][:Tmin], alpha=0.1)
    plt.xlabel("Update Time")
    plt.ylabel("Norm of w")

    plt.show()

'''References: ChatGPT-4o'''
