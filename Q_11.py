#  Solving the four given systems using numpy.linalg.solve

import numpy as np 

# 1st system 

A1 = np.array([[3,-1,1],[3,6,2],[3,3,7]])
b1 = np.array([1,0,4])
x1 = np.linalg.solve(A1, b1)

# solution for first system
print("solution for first system: ",x1)

# 2nd system 

A2 = np.array([[10,-1,0],[-1,10,-2],[0,-2,10]])
b2 = np.array([9,7,6])
x2 = np.linalg.solve(A2, b2)

# solution for second system
print("solution for second system: ",x2)

# 3rd system 

A3 = np.array([[10,5,0,0], [5,10,-4,0], [0,-4,8,-1],[0,0,-1,5]])
b3 = np.array([6,25,-11,-11])
x3 = np.linalg.solve(A3, b3)

# solution for third system
print("solution for third system: ",x3)

# 4th system 

A4 = np.array([[4,1,1,0,1], [-1,-3,1,1,0], [2,1,5,-1,-1], [-1,-1,-1,4,0], [0,2,-1,1,4]])
B4 = np.array([6,6,6,6,6])
x4 = np.linalg.solve(A4, B4)

# solution for fourth system
print("solution for fourth system: ",x4)