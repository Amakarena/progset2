# Let's use a single source file to upload the progset
import numpy as np

# A and B are nxn matrices (how to handle odd case? Add extra row
# of 0s and column of 0s. Delete extras after calculation and before
# output)
def conventional(A, B):
    # make lists
    A = A.tolist()
    B = B.tolist()
    # init output matrix
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    # by row of A
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                # calculate C
                C[i][j] += A[i][k] * B[k][j]
    return np.array(C)
    # print(np.matrix(C))

# pad function for odd n functions
def pad(A):
    n = len(A)
    padded_size = n + 1
    padded = np.zeros((padded_size, padded_size), dtype=int)
    padded[:n, :n] = A
    return padded

def strassen(X, Y, crossover = 2):
    n = len(X)
    # flag to remove padding after multiplication
    padded = False
    # check if n is odd and pad if need be
    if n % 2 != 0:
       X = pad(X)
       Y = pad(Y)
       n += 1
       padded = True

    # base case
    if n <= crossover:
        Z = conventional(X, Y)
        # removing excess 0s
        # if padded:
        #     # delete last row and column
        #     Z = Z[:-1, :-1]
        return Z

    # split them
    # A, B, C, D = split_matrix(X)
    # E, F, G, H = split_matrix(Y)
    mid = n // 2
    A = X[:mid, :mid]
    B = X[:mid, mid:]
    C = X[mid:, :mid]
    D = X[mid:, mid:]
    E = Y[:mid, :mid]
    F = Y[:mid, mid:]
    G = Y[mid:, :mid]
    H = Y[mid:, mid:]
    
    # mini multiplications
    p1 = strassen(A,np.subtract(F,H))
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

    # remove padding
    # Z = Z[:-1, :-1]

    return Z


# # ----- TESTS -----
# # coonventional
# X = np.array([[1,2,3],[3,4,3],[1,2,3]])
# Y = np.array([[5,6,3],[7,8,3],[1,2,3]])
# # X = np.array([[1,2],[3,4]])
# # Y = np.array([[5,6],[7,8]])
# # print(conventional(X,Y))

# # padding
# # D = [[1 for o in range(3)] for p in range(3)]
# # D = pad(D)

# # # splitting
# # a, b, c, d = split_matrix(D)
# # print(np.matrix(b))

# # # stacking
# # Z = np.vstack((np.hstack((a, b)), np.hstack((c, d))))
# # print(np.matrix(Z))

# # strassen
print(strassen(X, Y))