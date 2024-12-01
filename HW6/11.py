from libsvm.svmutil import *
from tqdm import tqdm
import numpy as np

data_file = "./mnist.scale"
y, X = svm_read_problem(data_file)

filtered_indices = [i for i, label in enumerate(y) if label in [3, 7]]
y = [1 if y[i] == 3 else -1 for i in filtered_indices]
X = [X[i] for i in filtered_indices]

Cs = [0.1, 1, 10]
gammas = [0.1, 1, 10]

results = []

def dis2(sparse_vec1, sparse_vec2):
    all_keys = set(sparse_vec1.keys()).union(sparse_vec2.keys())
    return sum((sparse_vec1.get(k, 0) - sparse_vec2.get(k, 0)) ** 2 for k in all_keys)


for i in tqdm(range(3), desc="Cs"):
    C = Cs[i]
    for gamma in gammas:
        param = f"-s 0 -t 2 -c {C} -g {gamma} -q"
        model = svm_train(y, X, param)

        sv_coef = model.get_sv_coef()
        sv_indices = model.get_sv_indices()
        support_vectors = [X[i - 1] for i in sv_indices]

        w_norm_squared = 0
        for i, alpha_i in enumerate(sv_coef):
            for j, alpha_j in enumerate(sv_coef):
                d2 = dis2(support_vectors[i], support_vectors[j])
                w_norm_squared += alpha_i[0] * alpha_j[0] * y[sv_indices[i]-1] * y[sv_indices[j]-1] * np.exp(-gamma * d2)

        margin = 1 / np.sqrt(w_norm_squared)
        results.append((C, gamma, margin))

print("C, Gamma, Margin")
for result in results:
    print(result)
