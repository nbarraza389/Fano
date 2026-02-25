# Casos de contagio

# Multiplicativo

beta = 2.0  # fuerza del contagio (ajustable)

z = np.zeros((c, n_X), dtype=int)

for y_prev in range(c):
    base = (X_states[:, 0] + X_states[:, 2]) % c
    z[y_prev] = (base + beta * y_prev) % c


# Campos

z = np.zeros((c, n_X), dtype=int)

for y_prev in range(c):
    field_X = (X_states[:, 0] + X_states[:, 2]) % c
    field_prev = y_prev * np.ones(n_X, dtype=int)

    z[y_prev] = (field_X + field_prev) % c

# Puro contagio

z = np.zeros((c, n_X), dtype=int)

for y_prev in range(c):
    z[y_prev] = y_prev

