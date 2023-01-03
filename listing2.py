# ************ Projected gradient method **********
# In presence of constraints, we can apply the projected gradient method.
# Firstly, we need to define the projection operator on a closed convex set, X
# The projected gradient method is
# x^(ν+1)=proj_x(x^ν−γ∇f(xν)) where, again,  0<γ≤1/L.

import numpy as np
import matplotlib.pyplot as plt

n = 30
W = np.random.standard_normal(size=(n, n))
Q = W.T @ W #Question 2: is this expression to define Q is a symmetric matrix?
q = np.random.standard_normal(size=(n, 1)) # a random vector

# The Lipschitz-constant of the gradient of f is
L = np.linalg.norm(Q) # L = ∥Q∥
gamma = 1 / L

# Let us define the gradient of f
def grad_f(x): # ∇f(x)
    return Q @ x + q

# define the projection on a ball of radius r
def projection_on_ball(x, radius):
    norm_x = np.linalg.norm(x)
    if norm_x <= radius:
        return x
    else:
        return (radius/ norm_x) * x

r = 12

# Numerical algorithm

x = np.zeros((n, 1))
n_iters = 5000

error_cache = []
for i in range(n_iters):
    df = grad_f(x)
    x_new = projection_on_ball(x - gamma * df, r)
    error = np.linalg.norm(x_new - x, np.inf)
    error_cache += [error]
    x = x_new
    if error < 1e-4:
        break

plt.semilogy(error_cache)
plt.xlabel('Iteration')
plt.ylabel('Inf. Norm of Gradient')
plt.title('Constrained QP, Projected Gradient Method')
plt.grid()
plt.show()
