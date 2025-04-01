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
    # divide into two
    upper = np.hsplit(np.vsplit(A, 2)[0], 2)
    lower = np.hsplit(np.vsplit(A, 2)[1], 2)
    # divide into four
    upper_left = upper[0]
    upper_right = upper[1]
    lower_left = lower[0]
    lower_right = lower[1]

    return upper_left, upper_right, lower_left, lower_right
    # print(np.matrix(A))
def strassen(X, Y, n_0):
    n = len(X)
    # flag to remove padding after multiplication
    padded = False
    # check if n is odd and pad if need be
    if n % 2 != 0:
       X = pad(X)
       Y = pad(Y)
       padded = True

    # split them
    A, B, C, D = split_matrix(X)
    E, F, G, H = split_matrix(Y)
    
    # mini multiplications
    p1 = strassen(A,(F-H))
    p2 = strassen((A+B),H)
    p3 = strassen((C+D),E)
    p4 = strassen(D,(G-E))
    p5 = strassen((A+D),(E+H))
    p6 = strassen((B-D),(G+H))
    p7 = strassen((C-A),(E+F))

    # calculating quadrants
    UL = p4 - p2 + p5 + p6
    UR = p1 + p2
    LL = p3 + p4
    LR = p1 - p3 + p5 + p7

    # stiching quads back together
    Z = np.vstack((np.hstack((UL, UR)), np.hstack((LL, LR))))

    # removing excess 0s
    if padded:
        # delete last row
        Z = np.delete(Z, (len(Z)-1), 0)
        # delete last column
        Z = np.delete(Z, len(Z)-1, 1)
    return Z


# def recursive_matrix_mult(n_matrix):

# ----- TESTS -----
# coonventional
X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
# print(conventional(X,Y))

# padding
D = [[1 for o in range(3)] for p in range(3)]
D = pad(D)

# splitting
a, b, c, d = split_matrix(D)
print(np.matrix(b))

# stacking
Z = np.vstack((np.hstack((a, b)), np.hstack((c, d))))
print(np.matrix(Z))

# print(np.matrix(D))
# print("Padded D")
# D = np.r_[D,[[0,0,0]]]
# D = np.c_[D, [0,0,0,0]]
# print(np.matrix(D))