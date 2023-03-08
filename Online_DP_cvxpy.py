import numpy as np
import cvxpy as cp

# Define parameters
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5, 0], [0, 0.5]])
F_u = np.array([[1, 0], [0, 1]])
Q = np.array([[1, 0], [0, 1]])
qN = np.array([0, 0])
R = np.array([[1, 0], [0, 1]])
r = np.array([0, 0])

# Online DP
N = 5
yN = np.zeros((2, N))
yN[:, 0] = np.array([1, 1])
P0 = Q
p0 = qN + F_u.T @ yN[:, -1]
uN = np.zeros((2, N-1))

# compute optimal control for each time step
for n in range(1, N):
    # compute R_tilde, K1, and d1
    Pt = P0 + A.T @ Q @ A - A.T @ P0 @ B @ np.linalg.solve(R + B.T @ P0 @ B, B.T @ P0 @ A)
    R_tilde = R + B.T @ Pt @ B
    K1 = -np.linalg.solve(R_tilde, B.T @ A)
    f = yN[:, n - 1]
    d1 = -np.linalg.solve(R_tilde, F_u.T @ yN[:, n - 1] + r + B.T @ (P0 @ f + p0))

    # Optimal control
    uN[:, n-1] = K1 @ yN[:, n - 1] + d1

    # Optimal state
    yN[:, n] = A @ yN[:, n - 1] + B @ uN[:, n-1]

    # update P0 and p0
    P0 = Pt
    p0 = qN + F_u.T @ yN[:, n]

print("yN:", yN)
print("uN:", uN)

# Solve using cvxpy
# Define the optimization variables x and u. x is a 2xN matrix representing the state at each time step, 
# and u is a 2x(N-1) matrix representing the control at each time step.
x = cp.Variable((2, N))
u = cp.Variable((2, N-1))

# Define the cost function
# The cost is the sum of squares of the state at the last time step
# and the sum of squares of the control input.
cost = cp.sum_squares(Q @ x[:, N-1]) + cp.sum_squares(R @ u)

# Define the constraints
# initial state x[:,0] is equal to [1, 1]
constr = [x[:,0] == np.array([1, 1])]
for n in range(N-1):
    constr += [x[:,n+1] == A @ x[:,n] + B @ u[:,n]]
    constr += [u[:,n] == -K1 @ x[:,n] - d1]

# create a Problem object with the cost and constraint functions and 
# call the solve() method to obtain the optimal values of x and u.
prob = cp.Problem(cp.Minimize(cost), constr)
prob.solve()

print("x_opt:", x.value)
print("u_opt:", u.value)

# the output should be the same as the yN and uN obtained from the online DP
