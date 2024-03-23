# Dominating eigenvalue and corresponding eigenvector using power methods
import numpy as np 

# Given matrix
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

# Initial guess 
x0 = np.ones(len(A))

# Tolerance
tolerance = 0.001   # eigenvalue accuracy 0.1 percent 
# power methods 
def power_method(A, x0, tolerance):
    err = 1
    iteration = 0
    while err >= tolerance:
        x = np.dot(A, x0)
        Ax = np.dot(A, x)
        eigenvalue_new = (np.dot(Ax.T, x))/(np.dot(x.T, x))
        if iteration == 0: 
            err = 1
        else: 
            err = np.abs(eigenvalue_new - eigenvalue)
        eigenvalue = eigenvalue_new
        x0 = x
        iteration += 1
    eigenvector = x/(np.linalg.norm(x))
    return eigenvalue, eigenvector, iteration

dominant_eigenvalue, eigenvector, iteration = power_method(A, x0, tolerance)
print("Dominant eigenvalue using power method: ", dominant_eigenvalue)
print("Corresponding eigenvector: ", eigenvector)
print("Iterations are required to reach 0.1 percent accuracy: ", iteration)


