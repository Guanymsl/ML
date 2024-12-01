from libsvm.svmutil import *
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

data_file = "./mnist.scale"
y, X = svm_read_problem(data_file)

filtered_indices = [i for i, label in enumerate(y) if label in [3, 7]]
y = [1 if y[i] == 3 else -1 for i in filtered_indices]
X = [X[i] for i in filtered_indices]

C = 1
gammas = [0.01, 0.1, 1, 10, 100]
num_iterations = 128

gamma_selection_counts = Counter()

for _ in tqdm(range(num_iterations), desc="Iterations"):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    val_indices = indices[:200]
    train_indices = indices[200:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    best_gamma = None
    best_error = float('inf')

    for gamma in gammas:
        param = f'-s 0 -t 2 -c {C} -g {gamma} -q'
        model = svm_train(y_train.tolist(), X_train.tolist(), param)
        _, p_acc, _ = svm_predict(y_val.tolist(), X_val.tolist(), model, '-q')
        error = 100 - p_acc[0]

        if error < best_error or (error == best_error and (best_gamma is None or gamma < best_gamma)):
            best_gamma = gamma
            best_error = error

    gamma_selection_counts[best_gamma] += 1

plt.bar(gamma_selection_counts.keys(), gamma_selection_counts.values())
plt.xlabel('Gamma')
plt.ylabel('Selection Frequency')
plt.title('Selection Frequency of Gamma Values (libsvm)')
plt.show()
