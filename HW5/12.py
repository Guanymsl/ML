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

for seed in tqdm(range(NUM), desc=f"Experiments"):
    random.seed(seed)
    results_summary = {}

    for lam in LAMS:
        C = 1 / lam

        param = f'-s 6 -c {C} -v 3 -q'

        E_cv = 1 - (train(y_train, X_train, param) / 100)
        results_summary[lam] = E_cv

    lambda_star = min(results_summary, key=lambda x: results_summary[x])
    if sum(v == results_summary[lambda_star] for v in results_summary.values()) > 1:
        lambda_star = max(l for l, v in results_summary.items() if v == results_summary[lambda_star])

    C = 1 / lambda_star

    param = f'-s 6 -c {C} -q'
    model = train(y_train, X_train, param)

    _, p_test_acc, _ = predict(y_test, X_test, model, options='-q')
    E_out_star = 1 - (p_test_acc[0] / 100)

    E_outs.append(E_out_star)

csv_filename = '12.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['E_out'])
    writer.writerows([[e] for e in E_outs])
