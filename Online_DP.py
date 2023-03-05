import numpy as np

# Define parameters
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5, 0], [0, 0.5]])
F_u = np.array([[1, 0], [0, 1]])
Q = np.array([[1, 0], [0, 1]])
qN = np.array([0, 0])
R = np.array([[1, 0], [0, 1]])
r = np.array([0, 0])

# initialise online DP
N = 3
yN = np.zeros((2, N))
yN[:, 0] = np.array([1, 1])
P0 = Q
p0 = qN + F_u.T @ yN[:, -1]

# compute optimal control for each time step
for n in range(1, N):
    # compute R_tilde, K1, and d1
    Pt = P0 + A.T @ Q @ A - A.T @ P0 @ B @ np.linalg.solve(R + B.T @ P0 @ B, B.T @ P0 @ A)
    R_tilde = R + B.T @ Pt @ B
    K1 = -np.linalg.solve(R_tilde, B.T @ A)
    f = yN[:, n-1]
    d1 = -np.linalg.solve(R_tilde, F_u.T @ yN[:, n-1] + r + B.T @ (P0 @ f + p0))

    # Optimal control
    u2_star = K1 @ yN[:, n-1] + d1

    # Optimal state
    yN[:, n] = A @ yN[:, n-1] + B @ u2_star

    # update P0 and p0
    P0 = Pt
    p0 = qN + F_u.T @ yN[:, n]

print(yN)
print(u2_star)
