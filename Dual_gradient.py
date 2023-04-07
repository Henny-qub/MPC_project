import numpy as np
import cvxpy as cp

# -- Problem data

# System matrices
A = np.array([[0.8, 1.2], [0, 0.5]])
B = np.array([[0.2],[0.1]])
f = np.array([[1],[-0.2]])

# Stage cost weights
Q = np.array([[1.5, 0],[0, 2.8]])
R = np.array([[5]])
q = np.array([[0.9], [-0.4]])
r = np.array([[-3]])

# Terminal cost weights
QN = 20 * np.eye(2)
qN = np.array([[2],[1]])

# State/input constraints
Fx = np.vstack((np.eye(2), -np.eye(2), np.zeros((2,2))))
Fu = np.vstack((np.zeros((4,1)), 1, -1))

# Terminal constraints
FN = np.eye(2)

# Prediction horizon
N = 3

# Dual vectors
ys = np.ones((6, 1, N))  # ys = (y0, y1, ..., yN-1)
for i in range(N):
    ys[:, :, i] = (i+1) * np.ones((6, 1))

yN = 4.567 * np.ones((2, 1))

# Initial condition
x_init = np.array([1, -2])

x_cvx = cp.Variable((2, N+1))
u_cvx = cp.Variable((1, N))

cost = 0
constr = [x_cvx[:, 0] == x_init]
for t in range(N):
    xt = x_cvx[:, t]
    ut = u_cvx[:, t]
    cost += ys[:, :, t].T @ (Fx @ xt + Fu @ ut)
    cost += 0.5 * (cp.quad_form(xt, Q) + cp.quad_form(ut, R))
    cost += q.T @ xt + r.T @ ut
    xnext = A @ xt + B @ ut + f.reshape((2,))
    constr += [x_cvx[:, t+1] == xnext]

cost += 0.5 * cp.quad_form(x_cvx[:, N], QN) + qN.T @ x_cvx[:, N]
cost += yN.T @ FN @ x_cvx[:, N]

prob = cp.Problem(cp.Minimize(cost), constr)
prob.solve()

print(u_cvx.value)
print(x_cvx.value)

# Let us check the result at t = N-1
P0 = QN
p0 = FN.T @ yN + qN
R_til_1 = R + B.T @ P0 @ B
K1 = -np.linalg.solve(R_til_1, B.T @ P0 @ A)
d1 = -np.linalg.solve(R_til_1, r + B.T @ (P0 @ f + p0) + Fu.T @ ys[:, :, N-1])

u_star_last = K1 @ x_cvx.value[:, N-1] + d1
error = np.abs(u_star_last - u_cvx[:, N-1].value)
assert error <= 1e-12, "Ooops! (1)"

# Let us check the result at t = N-2
Abar1 = A + B @ K1
P1 = Q + K1.T @ R @ K1 + Abar1.T @ P0 @ Abar1
p1 = (Fx + Fu @ K1).T @ ys[:, :, N-1] + K1.T @ (R @ d1 + r) + q + Abar1.T @ (P0 @ (B @ d1 + f) + p0)

R_til_2 = R + B.T @ P1 @ B
K2 = -np.linalg.solve(R_til_2, B.T @ P1 @ A)
d2 = -np.linalg.solve(R_til_2, r + B.T @ (P1 @ f + p1) + Fu.T @ ys[:, :, N-2])


u_star_penult = K2 @ x_cvx.value[:, N-2] + d2
error2 = u_star_penult - u_cvx[:, N-2].value
assert error2 <= 1e-12, "Ooops! (2)"
