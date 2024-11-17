import numpy as np
import random
import csv
from liblinear.liblinearutil import *
from tqdm import tqdm

train_file = './mnist.scale'
test_file = './mnist.scale.t'

y_train, X_train = svm_read_problem(train_file)
y_test, X_test = svm_read_problem(test_file)

train_mask = [(y == 2 or y == 6) for y in y_train]
test_mask = [(y == 2 or y == 6) for y in y_test]

y_train = [1 if y == 6 else -1 for y, m in zip(y_train, train_mask) if m]
X_train = [x for x, m in zip(X_train, train_mask) if m]
y_test = [1 if y == 6 else -1 for y, m in zip(y_test, test_mask) if m]
X_test = [x for x, m in zip(X_test, test_mask) if m]

LAMS = [10**i for i in range(-2, 4)]
NUM = 1126

E_outs = []
non_zero_counts = []

for seed in tqdm(range(NUM), desc="Experiments"):
    random.seed(seed)
    results_summary = {}

    for lam in LAMS:
        C = 1 / lam

        param = f'-s 6 -c {C} -q'
        model = train(y_train, X_train, param)

        _, p_train_acc, _ = predict(y_train, X_train, model, options='-q')
        E_in = 1 - (p_train_acc[0] / 100)

        _, p_test_acc, _ = predict(y_test, X_test, model, options='-q')
        E_out = 1 - (p_test_acc[0] / 100)

        non_zero_count = np.sum(np.array(model.get_decfun()[0]) != 0)

        results_summary[lam] = (E_in, E_out, non_zero_count)

    lambda_star = min(results_summary, key=lambda x: results_summary[x][0])
    if sum(v[0] == results_summary[lambda_star][0] for v in results_summary.values()) > 1:
        lambda_star = max(l for l, v in results_summary.items() if v[0] == results_summary[lambda_star][0])
    _, E_out_star, non_zero_count_star = results_summary[lambda_star]

    E_outs.append(E_out_star)
    non_zero_counts.append(non_zero_count_star)

csv_filename = f'10.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['E_out', 'Non_zero_components'])
    writer.writerows(zip(E_outs, non_zero_counts))
