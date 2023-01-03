import numpy as np
import matplotlib.pyplot as plt

# ************Gradient method**************
# We want to solve Minimise f(x) when x∈R^n
# We start by defining the problem data
# Q to be a random n by n positive semidefinite matrix
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

# We are now ready to solve the problem
x = np.zeros((n, 1))  # initial guess
n_iters = 50000

error_cache = []
for i in range(n_iters):
    df = grad_f(x)
    error = np.linalg.norm(df, np.inf) # error : ∥∇f(x*)∥∞
    error_cache += [error]
# next x = previous x - (gamma * ∇f(previous x)
    x = x - gamma * df # x^(ν+1)=x^ν−γ∇f(xν)
    if error < 1e-3:
        break
    # TODO: Make this loop stop as soon as the error
    #       becomes small enough

plt.semilogy(error_cache)
plt.xlabel('Iteration')
plt.ylabel('Inf. Norm of Gradient')
plt.title('Quadratic problem, Gradient Method')
plt.grid()
plt.show()
