import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(123)

# Parámetros
d = 5          # atributos
K = 3          # cardinalidad de cada atributo
c = 5          # clases
alpha = 0.3    # dificultad
N = 50_000     # tamaño del dataset

# Todos los posibles X
X_states = np.array(list(product(range(K), repeat=d)))
n_X = len(X_states)

P_Y_given_Yprev_X = np.full((c, n_X, c), alpha / (c - 1))

z = np.zeros((c, n_X), dtype=int)

for y_prev in range(c):
    z[y_prev] = (y_prev + X_states[:, 0] + X_states[:, 2]) % c

for y_prev in range(c):
    for x in range(n_X):
        y_star = z[y_prev, x]
        P_Y_given_Yprev_X[y_prev, x, y_star] = 1 - alpha

# muestreamos X i.i.d.
X_idx = rng.integers(0, n_X, size=N)

Y_samples = np.zeros(N, dtype=int)

# inicialización
Y_samples[0] = rng.integers(0, c)

for i in range(1, N):
    y_prev = Y_samples[i-1]
    x = X_idx[i]
    Y_samples[i] = rng.choice(c, p=P_Y_given_Yprev_X[y_prev, x])

#Entropía condicional teórica

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

H_Y_given_Yprev_X = np.mean([
    entropy(P_Y_given_Yprev_X[y, x])
    for y in range(c)
    for x in range(n_X)
])

print("H(Y_i | Y_{i-1}, X_i) =", H_Y_given_Yprev_X)

#Calculamos Fano

def fano_rhs(pe, c):
    if pe == 0 or pe == 1:
        return 0
    return entropy(np.array([pe, 1 - pe])) + pe * np.log2(c - 1)

pes = np.linspace(1e-4, 0.999, 10_000)
rhs = np.array([fano_rhs(pe, c) for pe in pes])

Pe_min = pes[rhs >= H_Y_given_Yprev_X][0]
print("Cota de Fano (Pe ≥):", Pe_min)

