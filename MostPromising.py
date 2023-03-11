import numpy as np
import cvxpy as cp

# Define parameters
# Dimensions
nx = 2 # state
nu = 1 # control
nc = 6 # constraints
N = 3
# System dynamics
A = np.array([[1, 1], [0, -1]])
B = np.array([[0.5], [0.1]])
f = np.array([[1], [0]])
# Constraints
Fx = np.vstack((np.eye(nx), -np.eye(nx), np.zeros((2, 2))))
Fu = np.vstack((np.zeros((4, 1)), 1, -1))
# Weights of stage cost function
Q = np.eye(nx)
R = np.array([[3]])
q = np.ones((nx, 1))
r = np.array([[0]])
# Terminal weights
QN = 10 * np.eye(nx)
qN = np.array([[0], [0]])
# Terminal constraints
FN = np.eye(nx)
# y's
y = np.ones((nc, N))
yN =  0 * np.ones((2, 1))

def dp(A, B, Fx, Fu, Q, R, q, r, QN, qN, FN, y, yN, N):
    # dimensions
    nx = A.shape[0]
    nc = y.shape[0]
    nu = B.shape[1]

    # allocating memory for caches
    P_cache = np.zeros((nx, nx, N+1))
    p_cache = np.zeros((nx, 1, N+1))
    R_tilde_cache = np.zeros((nu, nu, N))
    RF_cache = np.zeros((nu, nc, N))
    K_cache = np.zeros((nu, nx, N+1))
    d_cache = np.zeros((nu, 1, N+1))
    j_cache = np.zeros((nu, 1, N))
    A_bar_cache = np.zeros((nx, nx, N+1))
    Z_cache = np.zeros((nc, nx, N))
    s_cache = np.zeros((nx, 1, N))

    # Initialisation
    P_cache[:, :, 0] = QN
    p_cache[:, :, 0] = qN + FN.T @ yN

    # dynamic programming loop
    for t in range(N):
        P = P_cache[:, :, t]
        p = p_cache[:, :, t]
        Rtilde = R + B.T @ P @ B
        R_tilde_cache[:, :, t] = Rtilde
        bpa_temp = B.T @ P @ A
        K = -np.linalg.solve(Rtilde, bpa_temp)
        K_cache[:, :, t+1] = K
        RF_cache[:, :, t] = -Rtilde @ Fu.T
        j_cache[:, :, t] = -Rtilde @ (r + B.T@(P @ f + p))
        # y_current = y[:, N-t-1].reshape((nc, 1))
        y_current = y[:, t].reshape((nc, 1)) # added
        d = RF_cache[:, :, t] @ y_current + j_cache[:, :, t]
        d_cache[:, :, t+1] = d
        Abar = A + B @ K_cache[:, :, t+1]
        A_bar_cache[:, :, t+1] = Abar
        P_cache[:, :, t+1] = Q + K.T @ R @ K + Abar.T @ P @ Abar
        Z_cache[:, :, t] = Fx + Fu @ K
        # s_cache[:, :, t] = K.T @ (d + r) + q + Abar.T @ (B @ d + f + p)
        s_cache[:, :, t] = K.T @ (d + RF_cache[:, :, t] @ y_current) + q + Abar.T @ (B @ d + f + p) # added
        p_cache[:, :, t+1] = Z_cache[:, :, t].T @ y_current + s_cache[:, :, t]

    return {
        "P": P_cache,
        "K": K_cache,
        "d": d_cache,
        "Abar": A_bar_cache,
        "s": s_cache # added
    }


res = dp(A, B, Fx, Fu, Q, R, q, r, QN, qN, FN, y, yN, N)


# # Solve using cvxpy
x_cvx = cp.Variable((2, N+1))
u_cvx = cp.Variable((N, 1))

x_init = np.array([1, 1])


cost = 0
constr = [x_cvx[:,0] == x_init]
for t in range(N):
    y_current = y[:, t].reshape((nc, 1))
    cost += y_current.T @ (Fx @ x_cvx[:, t] + Fu @ u_cvx[t])
    cost += 0.5 * (cp.quad_form(x_cvx[:, t], Q) + cp.quad_form(u_cvx[t], R)) + q.T @ x_cvx[:, t] + r.T @ u_cvx[t]
    x_next = A @ x_cvx[:, t] + B @ u_cvx[t] + f.reshape((nx,))
    constr += [x_cvx[:, t+1] == x_next]

cost += yN.T @ (FN @ x_cvx[:, N]) + 0.5 * cp.quad_form(x_cvx[:, N], QN) + qN.T @ x_cvx[:, N]

prob = cp.Problem(cp.Minimize(cost), constr)
prob.solve()

print("x_opt:", x_cvx.value)
print("u_opt:", u_cvx.value)

K1 = res["K"][:, :, 1]
d1 = res["d"][:, :, 1]

x2 = x_cvx[:, N-1].value
err = K1 @ x2 + d1 - u_cvx[N-1].value
print(f"err = {err}")
