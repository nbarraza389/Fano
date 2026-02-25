import numpy as np
from itertools import product
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)

# Parámetros generales
N = 5_000          # número de muestras
d = 5               # número de atributos
cY = 2              # Y binaria
K = 100              # número de potenciales (fases)

# Cardinalidad de cada atributo X_j
card_X = 3          # valores {0,1,2}

# Generamos todas las combinaciones posibles de X

X_states = np.array(list(product(range(card_X), repeat=d)))
n_X = len(X_states)

# Parámetros del potencial para cada fase
# shape: (K, d)
A = rng.normal(loc=0.0, scale=1.0, size=(K, d))

#Para fases bien separadas

A[1] += 2.0
A[2] -= 2.0

Z_samples = rng.integers(0, K, size=N)

# Elegimos estados de X al azar
X_idx = rng.integers(0, n_X, size=N)
X_samples = X_states[X_idx]

# Elegimos estados de X al azar
X_idx = rng.integers(0, n_X, size=N)
X_samples = X_states[X_idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


Y_samples = np.zeros(N, dtype=int)

for i in range(N):
    k = Z_samples[i]
    x = X_samples[i]

    field = np.dot(A[k], x)  # h_k(X)
    p = sigmoid(field)  # P(Y=1 | X, Z=k)

    Y_samples[i] = rng.random() < p

#Procedemos a calcular Fano

def P_Y_given_X(X_states, A, pi_Z):
    """
    X_states : (n_X, d)
    A        : (K, d)
    pi_Z     : (K,)   distribución de fases
    """
    n_X = X_states.shape[0]
    K = A.shape[0]

    P_Y1 = np.zeros(n_X)

    for i, x in enumerate(X_states):
        p = 0.0
        for k in range(K):
            field = np.dot(A[k], x)
            p_k = sigmoid(field)
            p += pi_Z[k] * p_k
        P_Y1[i] = p

    return P_Y1

def binary_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

P_X = np.ones(n_X) / n_X

def conditional_entropy_Y_given_X(X_states, A, pi_Z, P_X):
    P_Y1 = P_Y_given_X(X_states, A, pi_Z)
    H = binary_entropy(P_Y1)
    return np.sum(P_X * H)

def fano_lower_bound(H_Y_given_X, n_classes=2):
    return (H_Y_given_X - 1) / np.log2(n_classes)

pi_Z = np.ones(K) / K   # fases equiprobables

H_Y_given_X = conditional_entropy_Y_given_X(
    X_states, A, pi_Z, P_X
)


def inverse_binary_entropy(H_target):
    """
    Resuelve H(p) = H_target para p en [0, 0.5]
    """
    return brentq(
        lambda p: binary_entropy(p) - H_target,
        1e-12,
        0.5 - 1e-12
    )

def fano_bound_binary(H_Y_given_X):
    return inverse_binary_entropy(H_Y_given_X)

#fano_bound = fano_lower_bound(H_Y_given_X)

#print("H(Y|X) =", H_Y_given_X)
#print("Fano lower bound on Pe =", fano_bound)

Pe_fano = fano_bound_binary(H_Y_given_X)

print("H(Y|X) =", H_Y_given_X)
print("Fano bound (binary) =", Pe_fano)


X_train, X_test, y_train, y_test = train_test_split(
    X_samples, Y_samples,
    test_size=0.3,
    random_state=0,
    stratify=Y_samples
)


rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=0
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

error_rf = 1 - accuracy_score(y_test, y_pred)

print("Error Random Forest:", error_rf)
