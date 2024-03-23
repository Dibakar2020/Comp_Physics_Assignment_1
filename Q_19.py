import numpy as np
import time

# Define the matrices

A = np.array([[2, 1],
              [1, 0]])

B = np.array([[2, 1],
              [1, 0],
              [0, 1]])

C = np.array([[2, 1],
              [-1, 1],
              [1, 1],
              [2, -1]])

D = np.array([[1, 1, 0],
              [-1, 0, 1],
              [0, 1, -1],
              [1, 1, -1]])

E = np.array([[1, 1, 0],
              [-1, 0, 1],
              [0, 1, -1],
              [1, 1, -1]])

F = np.array([[0, 1, 1],
              [0, 1, 0],
              [1, 1, 0],
              [0, 1, 0],
              [1, 0, 1]])

# Compute SVD for each matrix and measure time

# Matrix A 
start_time1 = time.time()
U1, S1, Vt1 = np.linalg.svd(A)
end_time1 = time.time()

print("Time taken:", end_time1 - start_time1, "seconds")
print("U1: \n", U1)
print("S1: \n", S1)
print("Vt1: \n", Vt1)


# Matrix B 
start_time2 = time.time()
U2, S2, Vt2 = np.linalg.svd(B)
end_time2 = time.time()
print("Time taken:", end_time2 - start_time2, "seconds")
print("U2: \n", U2)
print("S2: \n", S2)
print("Vt2: \n", Vt2)

# Matrix C 
start_time3 = time.time()
U3, S3, Vt3 = np.linalg.svd(C)
end_time3 = time.time()
print("Time taken:", end_time3 - start_time3, "seconds")
print("U3: \n", U3)
print("S3: \n", S3)
print("Vt3: \n", Vt3)


# Matrix D 
start_time4 = time.time()
U4, S4, Vt4 = np.linalg.svd(D)
end_time4 = time.time()
print("Time taken:", end_time4 - start_time4, "seconds")
print("U4: \n", U4)
print("S4: \n", S4)
print("Vt4: \n", Vt4)


# Matrix E 
start_time5 = time.time()
U5, S5, Vt5 = np.linalg.svd(E)  
end_time5 = time.time()
print("Time taken:", end_time5 - start_time5, "seconds")
print("U5: \n", U5)
print("S5: \n", S5)
print("Vt5: \n", Vt5)


# Matrix F 
start_time6 = time.time()
U6, S6, Vt6 = np.linalg.svd(F)
end_time6 = time.time()
print("Time taken:", end_time6 - start_time6, "seconds")
print("U6: \n", U6)
print("S6: \n", S6)
print("Vt6: \n", Vt6)

