import numpy as np

# given matrix
A = np.array([[5, -2],
              [-2, 8]])

# QR decomposition 
Q, R = np.linalg.qr(A)

# Q and R of the matrix A
print(" The Q matrix: \n", Q, "\n The R matrix: \n", R)

# Eigenvalue of the matrix using QR decomposition 
def qr_decom(A, tolerance = 0.0001):
    v = np.eye(len(A))
    err = 1
    eigenvalue = np.zeros(len(A))
    while err >=tolerance: 
        Q, R = np.linalg.qr(A)
        A_new = np.dot(R, Q)
        eigenvalue_new = np.diagonal(A_new)
        v = np.dot(v, Q)    # column of v matrix are the eigen vector of A matrix
        A = A_new
        err = np.abs(np.linalg.norm(eigenvalue)-np.linalg.norm(eigenvalue_new))
        eigenvalue = eigenvalue_new
    return eigenvalue , v

# Eigenvalues and eigenvectors produced by QR decomposition
eigenvalues , eigenvector = qr_decom(A)
print("Eigenvalues produced by QR decomposition:", eigenvalues)

# Eigenvalues and eigenvectors produced by numpy.linalg.eigh
eigenvalues_eigh, eigenvectors_eigh = np.linalg.eigh(A)
print("Eigenvalues produced by numpy.linalg.eigh:", eigenvalues_eigh)
