#!/usr/bin/env python3
import sys
import math
import time
import random
import matplotlib.pyplot as plt

GLOBAL_THRESHOLD = 2

'''
********************************************************************************
Conventional Algorithm O(n^3) - no bueno
********************************************************************************
'''
def conventional_multiply(A, B, d):
    C = [[0 for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(d):
            for k in range(d):
                C[i][j] += A[i][k] * B[k][j]
    return C

'''
********************************************************************************
Strassen Algorithm - Works as a power of 2 (!! ONLY !!)
********************************************************************************
'''
# Helper function to add matrices in starren
def add_matrix(A, B):
    d = len(A)
    return [[A[i][j] + B[i][j] for j in range(d)] for i in range(d)]

# Helper function to subtract matrices in starren
def sub_matrix(A, B):
    d = len(A)
    return [[A[i][j] - B[i][j] for j in range(d)] for i in range(d)]

# Helper function to split matrices into halves
def split(A):
    mid = len(A) // 2
    a = [i[:mid] for i in A[:mid]]
    b = [i[mid:] for i in A[:mid]]
    c = [i[:mid] for i in A[mid:]]
    d_ = [i[mid:] for i in A[mid:]]
    return a, b, c, d_
        
# Helper function to combine final quadrants in strassen algo.
def combine(c1, c2, c3, c4):
    top = [r1 + r2 for r1, r2 in zip(c1, c2)]
    bottom = [r1 + r2 for r1, r2 in zip(c3, c4)]
    return top + bottom

# Strassen Algorithm, at long last
def strassen_multiply(A, B):
    d = len(A)
    if d <= GLOBAL_THRESHOLD:
        return conventional_multiply(A, B, d)
    # else
    a, b, c, d_ = split(A)
    e, f, g, h = split(B)
    
    p1 = strassen_multiply(a, sub_matrix(f, h))
    p2 = strassen_multiply(add_matrix(a, b), h)
    p3 = strassen_multiply(add_matrix(c, d_), e)
    p4 = strassen_multiply(d_, sub_matrix(g, e))
    p5 = strassen_multiply(add_matrix(a, d_), add_matrix(e, h))
    p6 = strassen_multiply(sub_matrix(b, d_), add_matrix(g, h))
    p7 = strassen_multiply(sub_matrix(c, a), add_matrix(e, f))
    
    c1 = add_matrix(sub_matrix(add_matrix(p5, p4), p2), p6)
    c2 = add_matrix(p1, p2)
    c3 = add_matrix(p3, p4)
    c4 = add_matrix(sub_matrix(add_matrix(p1, p5), p3), p7)
    
    return combine(c1, c2, c3, c4)

'''
********************************************************************************
Padding Generalization to arbitrary n - so it includes odds
********************************************************************************
'''
# Helper function to find the next power of 2 for an arbitrary n
def next_power_of_two(n):
    """Return the smallest power of 2 >= n."""
    if n == 0:
        return 1
    return 2 ** math.ceil(math.log2(n))

# Helper function that pads a non-power of 2 function
def pad_matrix(A, new_size):
    old_size = len(A)
    padded = []
    for row in A:
        padded.append(row + [0]*(new_size - old_size))
    for _ in range(new_size - old_size):
        padded.append([0]*new_size)
    return padded

# Helper function that unpads the padded new matrix
def unpad_matrix(M, original_size):
    return [row[:original_size] for row in M[:original_size]]

def strassen_multiply_gen(A, B):
    d = len(A)
    new_size = next_power_of_two(d)
    if new_size == d:
        return strassen_multiply(A, B)
    # ELSE let us pad A, B THEN multiply as we normally would and unpad final C
    A_padded = pad_matrix(A, new_size)
    B_padded = pad_matrix(B, new_size)
    C_padded = strassen_multiply(A_padded, B_padded)
    C = unpad_matrix(C_padded, d)
    return C

'''
********************************************************************************
FINDING BEST THRESHOLD WITH TIMING + Graphing
********************************************************************************
'''

def set_strassen_threshold(value):
    global GLOBAL_THRESHOLD
    GLOBAL_THRESHOLD = value

def time_conventional(A, B):
    n = len(A)
    start = time.perf_counter()
    _ = conventional_multiply(A, B, n)
    end = time.perf_counter()
    return end - start

def time_strassen(A, B, threshold):
    set_strassen_threshold(threshold)
    start = time.perf_counter()
    _ = strassen_multiply_gen(A, B)
    end = time.perf_counter()
    return end - start

def compare_strassen_vs_conventional():
    """
    Compare conventional vs. Strassen with 5 different thresholds,
    on matrix sizes [64, 128, 256, 512, 1024].
    Then plot the results as lines on a single chart.
    """
    crossover_candidates = [32, 64] #To change

    test_sizes = [64, 128, 256, 512, 1024] # TO change

    value_set = [0, 1, 2]

    conventional_times = []
    strassen_times = {thr: [] for thr in crossover_candidates}

    for size in test_sizes:
        # Generate random A, B (same for each threshold)
        A = [[random.choice(value_set) for _ in range(size)] for _ in range(size)]
        B = [[random.choice(value_set) for _ in range(size)] for _ in range(size)]

        # Time conventional
        conv_time = time_conventional(A, B)
        conventional_times.append(conv_time)

        # For each threshold, time Strassen
        for thr in crossover_candidates:
            st_time = time_strassen(A, B, thr)
            strassen_times[thr].append(st_time)

    # ------------------------------------------
    # Plot: x-axis = test_sizes, y-axis = times
    # One line for conventional, one line for each threshold
    # ------------------------------------------
    plt.figure()  # Single plot with multiple lines

    # Plot conventional
    plt.plot(test_sizes, conventional_times, label="Conventional")

    # Plot strassen lines
    for thr in crossover_candidates:
        plt.plot(test_sizes, strassen_times[thr], label=f"Strassen(thr={thr})")

    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Conventional vs. Strassen with various thresholds")
    plt.legend()

    plt.show()


'''
********************************************************************************
COUNTING TRIANGLE VIA A^3 (A^2 * A)
********************************************************************************
'''
N = 1024
GLOBAL_THRESHOLD = 64   # base-case dimension for Strassen
PROBS = [0.01, 0.02, 0.03, 0.04, 0.05]

# Building random adjacency
def build_random_adjacency(n, p):
    """
    Returns an n x n adjacency matrix for an undirected graph
    where each edge is included with probability p.
    No self-loops.
    """
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                A[i][j] = 1
                A[j][i] = 1
    return A

# Counting Triangle through strassen
def count_triangles(A):
    """
    1) Compute A^2 = A*A with Strassen
    2) Then compute A^3 = (A^2)*A
    3) #triangles = sum diag(A^3)/6
    """
    n = len(A)

    # A^2
    A2 = strassen_multiply_gen(A, A)
    # A^3
    A3 = strassen_multiply_gen(A2, A)

    # sum diagonal
    diag_sum = 0
    for i in range(n):
        diag_sum += A3[i][i]

    # each triangle counted 6 times in A^3 for an undirected graph
    return diag_sum // 6

'''
********************************************************************************
MAIN
********************************************************************************
'''
def main():
    compare_strassen_vs_conventional()
    
# def main():
#     # precompute binomial(1024, 3)
#     comb_1024_3 = math.comb(N, 3)

#     results = []
#     for p in PROBS:
#         print(f"\n=== p = {p} ===")
#         # build adjacency
#         start_build = time.perf_counter()
#         A = build_random_adjacency(N, p)
#         end_build = time.perf_counter()

#         print(f"Built adjacency in {end_build - start_build:.2f}s")

#         # count triangles
#         start_tri = time.perf_counter()
#         tri_count = count_triangles(A)
#         end_tri = time.perf_counter()

#         print(f"Counted triangles in {end_tri - start_tri:.2f}s")

#         # expected
#         expected = comb_1024_3 * (p**3)

#         print(f"Actual # triangles: {tri_count},  Expected ~ {expected:.1f}")

#         results.append((p, tri_count, expected))

#     # Example: just print the final table
#     print("\n===== Final Results =====")
#     print("p\tTriangles\tExpected")
#     for (p, tri_count, exp) in results:
#         print(f"{p}\t{tri_count}\t{exp:.1f}")


# -----------------------------
# MAIN
#   usage: ./strassen 0 dimension inputfile
#   read dimension d
#   read 2*d*d integers => build A, B
#   compute C = A*B
#   print diag
# -----------------------------

# def main():
#     if len(sys.argv) != 4:
#         print("Usage: {} <flag> <dimension> <inputfile>".format(sys.argv[0]))
#         sys.exit(1)

#     # parse
#     flag = sys.argv[1]           # e.g. '0'
#     d = int(sys.argv[2])         # dimension
#     inputfile = sys.argv[3]

#     # read file => 2*d*d integers
#     with open(inputfile, "r") as f:
#         nums = [int(line.strip()) for line in f if line.strip()]

#     if len(nums) != 2*d*d:
#         print("Error: input file does not have 2*d*d = {} integers.".format(2*d*d))
#         sys.exit(1)

#     # first d*d => matrix A
#     Anums = nums[:d*d]
#     # next d*d => matrix B
#     Bnums = nums[d*d:2*d*d]

#     # convert flat lists => 2D
#     A = [Anums[i*d:(i+1)*d] for i in range(d)]
#     B = [Bnums[i*d:(i+1)*d] for i in range(d)]

#     # multiply
#     # If you want to ensure it works for all d (not just power-of-2), do:
#     C = strassen_multiply_gen(A, B)
#     # else you can do:
#     # C = strassen_multiply(A, B)

#     # print diagonal c[0,0], c[1,1], ... c[d-1,d-1]
#     # one per line + trailing newline
#     for i in range(d):
#         print(C[i][i])

if __name__ == "__main__":
    main()