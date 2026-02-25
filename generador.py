from itertools import count

import numpy as np
import itertools
import random
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import beta

rng = np.random.default_rng(seed=42)

# Parámetros
d = 5          # número de atributos
k = 3          # cardinalidad de cada atributo
c = 5          # clases
N = 10000    # tamaño del dataset

# 1. Espacio de estados de X
X_states = list(itertools.product(range(k), repeat=d))
n_X = len(X_states)  # 3^5 = 243

# 2. P(X)
# P_X = rng.random(n_X) # Uniforme
# a=2, b=2 concentra los números en el centro (forma de campana)
# a=0.5, b=0.5 concentra los números cerca de 0 y 1
#P_X = beta.rvs(a=0.5, b=0.5, size=n_X) # Beta

#Triangular
# Parámetros: límite inferior, límite superior, moda
P_X = np.array([random.triangular(0, 1, 0.1) for _ in range(n_X)])

P_X /= P_X.sum()

# 3. P(Y|X) uniforme
#P_Y_given_X = rng.random((n_X, c))
#P_Y_given_X /= P_Y_given_X.sum(axis=1, keepdims=True)

# 4. Generación del dataset
X_samples = []
Y_samples = []

X_indices = rng.choice(n_X, size=N, p=P_X)

print("X_indices:", X_indices)

for idx in X_indices:
    x = X_states[idx]
#   y = rng.choice(c, p=P_Y_given_X[idx])
    X_samples.append(x)
#    Y_samples.append(y)


X_samples = np.array(X_samples)
Y_samples = np.array(Y_samples)

# X_samples: shape (N, 5), valores en {0,1,2}
#z = (X_samples[:,0] + X_samples[:,2]) % c
z = (np.array(X_states)[:,0] + np.array(X_states)[:,2]) % c

print("Len z",len(z))

alpha = 0.4  # controla dificultad (clave)

# Versión optimizada y más clara
P_Y_given_X = np.full((n_X, c), alpha / (c - 1)) # Llena todo con el valor pequeño
for i in range(n_X):
    y_star = z[i]
    P_Y_given_X[i, y_star] = 1 - alpha           # Sobrescribe la clase correcta


#idx = rng.integers(0, n_X, size=N)

Y_samples = np.array([
       rng.choice(c, p=P_Y_given_X[i])
       for i in X_indices])

#P_Y_given_X = np.zeros((n_X, c))

#for i in range(n_X):
#    y_star = z[i]
#    P_Y_given_X[i] = alpha / (c - 1)
#    P_Y_given_X[i, y_star] = 1 - alpha

print("Dataset generado:")
print("X shape:", X_samples.shape)
print("Y shape:", Y_samples.shape)
print(X_samples)
print(Y_samples)

# Estimar P(Y|X) empírico para un X fijo

caso = 200 #Fila del dataset, corresponde al caso de muestra
x0 = X_states[caso]
mask = np.all(X_samples == x0, axis=1)

print("Mask:", mask)
print(np.bincount(mask))

counts = Counter(Y_samples[mask])

print("counts", counts.values())

total = sum(counts.values())

empirical = np.array([counts[i]/total if total > 0 else 0 for i in range(c)])
true = P_Y_given_X[caso]

print("P(Y|X=x0) verdadero:", np.round(true, 3))
print("P(Y|X=x0) empírico:", np.round(empirical, 3))

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# P(Y)
P_Y = np.sum(P_X[:, None] * P_Y_given_X, axis=0)

H_Y = entropy(P_Y)

# H(Y|X)
H_Y_given_X = 0.0
for i in range(n_X):
    H_Y_given_X += P_X[i] * entropy(P_Y_given_X[i])

print("H(Y) =", H_Y)
print("H(Y|X) =", H_Y_given_X)

M = c  # número de clases

def fano_rhs(pe):
    if pe == 0 or pe == 1:
        return 0
    return entropy(np.array([pe, 1-pe])) + pe * np.log2(M-1)

pes = np.linspace(1e-6, 1-1e-6, 5000)
rhs = np.array([fano_rhs(pe) for pe in pes])

print("H_Y_given_X")
print(H_Y_given_X)
print(np.max(rhs), np.argmax(rhs))

#Pe_min = pes[rhs >= H_Y_given_X][0]

Pe_min = pes[np.round(rhs, 5) >= round(H_Y_given_X, 5)][0]

print("Cota inferior de P_e (Fano):", Pe_min)



# Recorro los alphas
# def generate_conditional(X_samples, alpha):
#    z = (X_samples[:,0] + X_samples[:,2]) % c
#    P = np.zeros((len(X_samples), c))
#    for i in range(len(X_samples)):
#        P[i] += alpha / (c - 1)
#        P[i, z[i]] = 1 - alpha
#    return P

#alphas = np.linspace(0.1, 0.9, 20)

#print("Elijo alpha")

#for alpha in alphas:
#    P_Y_given_X = generate_conditional(X_samples, alpha)
#    H_Y_given_X = np.mean([entropy(P_Y_given_X[i]) for i in range(n_X)])
#    # calcular Fano → Pe_min
#    print(alpha, H_Y_given_X)

X = X_samples
Y = Y_samples

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=0
)

rf.fit(X, Y)

Y_hat = rf.predict(X)

Pe_rf = 1 - accuracy_score(Y, Y_hat)

print("Error RF:", Pe_rf)
print("Cota Fano:", Pe_min)
