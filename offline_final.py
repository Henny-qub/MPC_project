import numpy as np
import cvxpy as cp

# Define parameters
# Dimensions
nx = 2 # state
nu = 1 # control
nc = 6
N = 3
# System dynamics
A = np.array([[1, 1], [0, -1]])
B = np.array([[0.5], [0.1]])
f = np.array([[0.5], [0]])
# Constraints
Fx = np.vstack((np.eye(nx), -np.eye(nx), np.zeros((2, 2))))
Fu = np.vstack((np.zeros((4, 1)), 1, -1))
# Weights of stage cost function
Q = np.eye(nx)
R = np.array([[3]])
q = np.ones((nx, 1))
r = np.array([[2]])
# Terminal weights
QN = 10 * np.eye(nx)
qN = np.array([[1], [2]])
# Terminal constraints
FN = np.eye(nx)
# y's
y = np.ones((nc, N))
yN =  1.5 * np.ones((2, 1))

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
    j_cache = np.zeros((nu, 1, N))
    d_cache = np.zeros((nu, 1, N))
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
        y_current = y[:, N-t-1].reshape((nc, 1))
        d = RF_cache[:, :, t] @ y_current + j_cache[:, :, t]
        d_cache[:, :, t] = d
        Abar = A + B @ K_cache[:, :, t+1]
        A_bar_cache[:, :, t+1] = Abar
        P_cache[:, :, t+1] = Q + K.T @ R @ K + Abar.T @ P @ Abar
        Z_cache[:, :, t] = Fx + Fu @ K
        s_cache[:, :, t] = K.T @ (d + r) + q + Abar.T @ (B@d + f + p)
        p_cache[:, :, t+1] = Z_cache[:, :, t].T @ y_current + s_cache[:, :, t]

    return P_cache


P_cache = dp(A, B, Fx, Fu, Q, R, q, r, QN, qN, FN, y, yN, N)
print(P_cache[:, :, 0])
    # # # Online DP
    # yN = y[:, N]
    # P0 = Q
    # p0 = qN + Fu.T @ yN
    # uN = np.zeros((2, N-1))
    #
    # # compute optimal control for each time step
    # for n in range(1, N):
    #     # compute R_tilde, K1, and d1
    #     Pt = P0 + A.T @ Q @ A - A.T @ P0 @ B @ np.linalg.solve(R + B.T @ P0 @ B, B.T @ P0 @ A)
    #     R_tilde = R + B.T @ Pt @ B
    #     K1 = -np.linalg.solve(R_tilde, B.T @ A)
    #     f = y[:, n - 1]
    #     d1 = -np.linalg.solve(R_tilde, Fu.T @ yN[:, n - 1] + r + B.T @ (P0 @ f + p0))
    #
    #     # Optimal control
    #     uN[:, n-1] = K1 @ y[:, n - 1] + d1
    #
    #     # Optimal state
    #     yN[:, n] = A @ y[:, n - 1] + B @ uN[:, n-1]
    #
    #     # update P0 and p0
    #     P0 = Pt
    #     p0 = qN + Fu.T @ yN[:, n]

# print("yN:", yN)
# print("uN:", uN)
#
# # Solve using cvxpy
# x = cp.Variable((2, N))
# u = cp.Variable((N-1, 1))



# cost = cp.sum_squares(Q @ x[:, N-1]) + cp.sum_squares(R @ u)
# constr = [x[:,0] == np.array([1, 1])]
# for n in range(N-1):
#     constr += [x[:,n+1] == A @ x[:,n] + B @ u[:,n]]
#     constr += [u[:,n] == -K1 @ x[:,n] - d1]
#
# prob = cp.Problem(cp.Minimize(cost), constr)
# prob.solve()

# print("x_opt:", x.value)
# print("u_opt:", u.value)
