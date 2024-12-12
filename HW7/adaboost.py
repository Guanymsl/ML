import numpy as np
from sklearn.metrics import zero_one_loss
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    X = []
    y = []

    for line in lines:
        entries = line.strip().split()
        y.append(float(entries[0]))
        features = []

        for entry in entries[1:]:
            _, value = entry.split(':')
            features.append(float(value))

        X.append(features)

    X = np.array(X)
    y = np.array(y)

    return X, y

def decision_stump(X, y, weights):
    m, n = X.shape
    best_stump = {}
    min_error = float('inf')

    for feature in range(n):
        feature_values = X[:, feature]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = polarity * np.sign(X[:, feature] - threshold)
                error = np.sum(weights[y != predictions]) / np.sum(weights)

                if error < min_error:
                    min_error = error
                    best_stump = {
                        'feature': feature,
                        'threshold': threshold,
                        'polarity': polarity
                    }
    return best_stump, min_error

def adaboost(X_train, y_train, X_test, y_test, T):
    m, n = X_train.shape
    weights = np.ones(m) / m

    alphas = []
    stumps = []

    E_in_g_t = []
    epsilon_t = []
    E_in_G_t = []
    E_out_G_t = []
    U_t = []

    for t in tqdm(range(T), desc="Iterations"):
        stump, error = decision_stump(X_train, y_train, weights)
        alpha = 0.5 * np.log((1 - error) / error)

        predictions = stump_predict(X_train, stump)

        stumps.append(stump)
        alphas.append(alpha)
        G_t_train = strong_classifier(X_train, stumps, alphas)
        G_t_test = strong_classifier(X_test, stumps, alphas)

        epsilon_t.append(error)
        E_in_g_t.append(zero_one_loss(y_train, predictions))
        E_in_G_t.append(zero_one_loss(y_train, G_t_train))
        E_out_G_t.append(zero_one_loss(y_test, G_t_test))
        U_t.append(np.sum(weights))

        weights = weights * np.exp(-alpha * y_train * predictions)

    return E_in_g_t, epsilon_t, E_in_G_t, E_out_G_t, U_t

def stump_predict(X, stump):
    feature = stump['feature']
    threshold = stump['threshold']
    polarity = stump['polarity']
    return polarity * np.sign(X[:, feature] - threshold)

def strong_classifier(X, stumps, alphas):
    m = X.shape[0]
    final_prediction = np.zeros(m)

    for alpha, stump in zip(alphas, stumps):
        final_prediction += alpha * stump_predict(X, stump)
    return np.sign(final_prediction)

if __name__ == "__main__":
    X_train, y_train = load_data('./data/madelon')
    X_test, y_test = load_data('./data/madelon.t')

    T = 500
    E_in_g_t, epsilon_t, E_in_G_t, E_out_G_t, U_t = adaboost(X_train, y_train, X_test, y_test, T)

    metrics = {
        'E_in_g_t': E_in_g_t,
        'epsilon_t': epsilon_t,
        'E_in_G_t': E_in_G_t,
        'E_out_G_t': E_out_G_t,
        'U_t': U_t
    }

    with open('./data/adaboost_metrics.txt', 'w') as f:
        for key, values in metrics.items():
            f.write(f"{key}:\n")
            f.write(" ".join(map(str, values)) + "\n")
