# Let's use a single source file to upload the progset
import numpy as np

#n_0 from theoretical approach
# n = 4

# # creating an n*n matrix 
# matrix = np.zeros((n, n), dtype=int)
# print(matrix)
# print(len(matrix))

# A and B are nxn matrices (how to handle odd case? Add extra row
# of 0s and column of 0s. Delete extras after calculation and before
# output)
def conventional(A, B):
    # init output matrix
    C = [[0 for l in range(len(A))] for m in range(len(A[0]))]
    # by row of A
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                # calculate C
                C[i][j] += A[i][k] * B[k][j]
    print(np.matrix(C))

def pad(A):
    A = np.r_[A,[[0 for i in range(len(A))]]]
    A = np.c_[A, [0 for i in range(len(A[0])+1)]]
    return A

# got from stack
def split_matrix(A):
    upper = np.hsplit(np.vsplit(A, 2)[0], 2)
    lower = np.hsplit(np.vsplit(A, 2)[1], 2)

    upper_left = upper[0]
    upper_right = upper[1]
    lower_left = lower[0]
    lower_right = lower[1]

    return upper_left, upper_right, lower_left, lower_right
    # print(np.matrix(A))
# def strassen(A, B, n_0):
#     C = 

# def recursive_matrix_mult(n_matrix):

# ----- TESTS -----
# coonventional
X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
print(conventional(X,Y))

# padding
D = [[1 for o in range(3)] for p in range(3)]
D = pad(D)

a, b, c, d = split_matrix(D)
print(np.matrix(b))

# print(np.matrix(D))
# print("Padded D")
# D = np.r_[D,[[0,0,0]]]
# D = np.c_[D, [0,0,0,0]]
# print(np.matrix(D))