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
    print(C)

X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
print(conventional(X,Y))

# def strassen(n_matrix):

# def recursive_matrix_mult(n_matrix):

# ----- TESTS -----