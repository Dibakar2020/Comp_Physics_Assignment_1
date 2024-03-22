import numpy as np

# Given matrix A
A = np.array([[0.2, 0.1, 1, 1, 0],
              [0.1, 4, -1, 1, -1],
              [1, -1, 60, 0, -2],
              [1, 1, 0, 8, 4],
              [0, -1, -2, 4, 700]])

# Given vector b
b = np.array([1, 2, 3, 4, 5])

# True solution
true_solution = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])

# Initial guess for the solution vector
x0 = np.zeros_like(b)

# Tolerance
tolerance = 0.01

# Jacobi method
def jacobi(A, b, x0, tolerance):
    n = len(b)
    x = x0.copy()
    iterations = 0
    err = 1
    while err >= tolerance:
        xx = []
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += (-A[i][j]*x[j])
            s += b[i]
            s /= A[i][i]
            xx.append(s)
        err = np.abs(np.linalg.norm(xx) - np.linalg.norm(x))
        x = xx
        iterations += 1
    return x, iterations

# Solve using Jacobi method
x_jacobi, iterations_jacobi = jacobi(A, b, x0, tolerance)

# Print results
print("Jacobi method:")
print("Solution:", x_jacobi)
print("Iterations:", iterations_jacobi)
        
# ------------------------------------------------------------------------
# Gauss-Seidel method( GS method )

def gauss_seidel_method(A, b, x0, tolerance, max_iterations = 100):

    x = np.zeros(len(x0))
    x_prev = x0

    for k in range(1, max_iterations):
        
        for i in range(len(A)):

            s2 = 0
            s3 = 0
            s1 = 0

            s1 = s1 +  b[i]/A[i][i] 

            for j1 in range(i):
                s2 = s2 - (A[i][j1]*x[j1])/ A[i][i]
            

            for j2 in range(i+1,len(A)):
                s3 = s3 - (A[i][j2]*x_prev[j2] )/ A[i][i]
            

            x[i] = s1 + s2 + s3 


        # setting up the tolerence
        if np.abs(np.linalg.norm(x - true_solution)) < tolerance:
            return x, k
        
        x_prev = x.copy()

    return x


# Solve using Jacobi method
gs_solution, gs_iteration = gauss_seidel_method(A, b, x0, tolerance)

# Print results
print("\nGauss-Seidel method:")
print("Solution:", gs_solution)
print("Iterations:", gs_iteration)


# ----------------------------------------------------------------------------
# Relaxation method
def relaxation_method(A, b, x0, tolerance, w , max_iterations = 100):
    x = np.zeros(len(x0))
    x_prev = x0

    for k in range(1,max_iterations):
        
        for i in range(len(A)):

            s2 = 0
            s3 = 0
            
            s1 = (1 - w) * x[i] + (w * b[i])/A[i][i]
            
            for j1 in range(i):
                s2 = s2 - A[i][j1]*x[j1] * w / A[i][i]
            

            for j2 in range(i+1,len(A)):
                s3 = s3 - A[i][j2]*x_prev[j2] * w / A[i][i]

            x[i] = s1 + s2 + s3
             
        

        if np.abs(np.linalg.norm(x - true_solution)) < tolerance:
            return x, k
        
        x_prev = x.copy()

    return x  

# Solve using Relaxation method
w = 1.25  # Relaxation parameter
relax_solution, relax_iteration = relaxation_method(A, b, x0, tolerance, w)

print("\nRelaxation method:")
print("Solution:", relax_solution)
print("Iterations:", relax_iteration)


# --------------------------------------------------------
# Conjugate Gradient method (CG method)
def conjugate_gradient_method(A, b, x0, tolerance, max_iterations = 100000):

    x = np.zeros(len(A))

    for k in range(1, max_iterations):

        r = b - np.dot(A, x)

        Ar = np.dot(A, r)

        t1 = np.dot(r.T, r)/ np.dot(r.T, Ar)

        x = x + t1*r

        if np.abs(np.linalg.norm(x - true_solution)) < tolerance:
            return x, k

    return x, k

# Solve using Conjugate Gradient method
cg_solution, cg_iteration = conjugate_gradient_method(A, b, x0, tolerance)

# Print the result from Conjugate Gradient method
print("\nConjugate Gradient method:")
print("Solution:", cg_solution)
print("Iterations:", cg_iteration)