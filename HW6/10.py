from libsvm.svmutil import *
from tqdm import tqdm

data_file = "./mnist.scale"
y, X = svm_read_problem(data_file)

filtered_indices = [i for i, label in enumerate(y) if label in [3, 7]]
y = [1 if y[i] == 3 else -1 for i in filtered_indices]
X = [X[i] for i in filtered_indices]

Cs = [0.1, 1, 10]
Qs = [2, 3, 4]

results = []

for i in tqdm(range(3), desc="Cs"):
    C = Cs[i]
    for Q in Qs:
        param = f"-s 0 -t 1 -c {C} -d {Q} -q -g 1 -r 1"
        model = svm_train(y, X, param)
        support_vectors_count = model.get_nr_sv()

        results.append((C, Q, support_vectors_count))

print("C, Q, Support Vector Count")
for result in results:
    print(result)
