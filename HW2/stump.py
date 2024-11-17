import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

np.random.seed(42)

N = 12
p = 0.15
iterations = 2000

Ein_list_original = []
Eout_list_original = []
Ein_list_modified = []
Eout_list_modified = []

def f(x):
    y = sign(x)
    flip = np.random.rand(len(y)) < p
    y_noisy = y.copy()
    y_noisy[flip] *= -1
    return y_noisy

def h(s, theta, x):
    return s * sign(x - theta)

def get_thetas(x):
    x_sorted = np.sort(x)
    thetas = [-1]
    for i in range(N - 1):
        if x_sorted[i] != x_sorted[i + 1]:
            theta = (x_sorted[i] + x_sorted[i + 1]) / 2
            thetas.append(theta)
    return thetas

def sign(x):
    return np.where(x <= 0, -1, 1)

def get_Ein(s, theta, x, y):
    return np.mean(h(s, theta, x) != y)

def original_decision_stump_model():
    for i in tqdm(range(iterations), desc="Iterations"):
        x = np.random.uniform(-1, 1, N)
        y = f(x)

        best_Ein = float('inf')
        best_s = None
        best_theta = None

        thetas = get_thetas(x)
        for s in [-1, 1]:
            for theta in thetas:
                Ein = get_Ein(s, theta, x, y)
                if Ein < best_Ein or (Ein == best_Ein and s * theta < best_s * best_theta):
                    best_Ein = Ein
                    best_s = s
                    best_theta = theta

        Eout = 0.5 - 0.5 * best_s + best_s * p + best_s * (0.5 - p) * np.abs(best_theta)

        Ein_list_original.append(best_Ein)
        Eout_list_original.append(Eout)

def modified_decision_stump_model():
    for i in tqdm(range(iterations), desc="Iterations"):
        x = np.random.uniform(-1, 1, N)
        y = f(x)

        s = np.random.choice([-1, 1])
        theta = np.random.uniform(-1, 1)

        Ein = get_Ein(s, theta, x, y)
        Eout = 0.5 - 0.5 * s + s * p + s * (0.5 - p) * np.abs(theta)

        Ein_list_modified.append(Ein)
        Eout_list_modified.append(Eout)

def plot():
    Ein_array_original = np.array(Ein_list_original)
    Eout_array_original = np.array(Eout_list_original)
    Ein_array_modified = np.array(Ein_list_modified)
    Eout_array_modified = np.array(Eout_list_modified)

    delta = Eout_array_original - Ein_array_original
    median_delta = np.median(delta)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Ein_array_original, y=Eout_array_original, color='blue')
    plt.title(f'Median of Eout - Ein (Original Model): {median_delta:.4f}')
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.grid(True)

    delta = Eout_array_modified - Ein_array_modified
    median_delta = np.median(delta)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Ein_array_modified, y=Eout_array_modified, color='orange')
    plt.title(f'Median of Eout - Ein (Modified Model): {median_delta:.4f}')
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Ein_array_original, y=Eout_array_original, alpha=0.5, color='blue', label='Original Model')
    sns.scatterplot(x=Ein_array_modified, y=Eout_array_modified, alpha=0.5, color='orange', label='Modified Model')
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Running Original Decision Stump Model...")
    original_decision_stump_model()
    print("Running Modified Decision Stump Model...")
    modified_decision_stump_model()
    plot()
