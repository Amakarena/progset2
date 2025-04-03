#!/usr/bin/env python3
import sys
import math
import time
import random
# import matplotlib.pyplot as plt

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
# Helper function to find the next power of 2 for an arbitrary n, check works w/ odd n
def next_power_of_two(n):
    if n == 0:
        return 1
    return 2 ** math.ceil(math.log2(n))

# Helper function that pads a non-power of 2 function if called
def pad_matrix(A, new_size):
    old_size = len(A)
    padded = []
    for row in A:
        padded.append(row + [0]*(new_size - old_size))
    for _ in range(new_size - old_size):
        padded.append([0]*new_size)
    return padded

# Helper function that unpads the padded new matrix at final output
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
# Changes global threshold when best one found
def set_strassen_threshold(value):
    global GLOBAL_THRESHOLD
    GLOBAL_THRESHOLD = value

#  Test time for Conventional
def time_conventional(A, B):
    n = len(A)
    start = time.perf_counter()
    _ = conventional_multiply(A, B, n)
    end = time.perf_counter()
    return end - start

#  Test time for Strassen
def time_strassen(A, B, threshold):
    set_strassen_threshold(threshold)
    start = time.perf_counter()
    _ = strassen_multiply_gen(A, B)
    end = time.perf_counter()
    return end - start

def compare_strassen_vs_conventional():
    # Threshold params. TO EXPERIMENT WITH
    crossover_candidates = [32, 64]
    # Dimension sizes. TO EXPERIMENT WITH 
    test_sizes = [257, 515, 1031, 2001] #Try even AND ODD

    value_set = [0, 1, 2]
    conventional_times = []
    strassen_times = {thr: [] for thr in crossover_candidates}
    for size in test_sizes:
        # Generate random A, B (same for each threshold)
        A = [[random.choice(value_set) for _ in range(size)] for _ in range(size)]
        B = [[random.choice(value_set) for _ in range(size)] for _ in range(size)]

        conv_time = time_conventional(A, B)
        conventional_times.append(conv_time)
        # For each threshold, time Strassen
        for thr in crossover_candidates:
            st_time = time_strassen(A, B, thr)
            strassen_times[thr].append(st_time)
    
    # SINGLE Plot w/ ALL algorithms plotted for comparison
    plt.figure()
    plt.plot(test_sizes, conventional_times, label="Conventional")
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
GLOBAL_THRESHOLD = 32   # base-case dimension for Strassen based on analysis above!
PROBS = [0.01, 0.02, 0.03, 0.04, 0.05]

# Building random adjacency matrix for undirect. graph
def build_random_adjacency(n, p):
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                A[i][j] = 1
                A[j][i] = 1
    return A

# Counting Triangle through strassen
def count_triangles(A):
    n = len(A)
    # A^2
    A2 = strassen_multiply_gen(A, A)
    # A^3
    A3 = strassen_multiply_gen(A2, A)
    
    diag_sum = 0
    for i in range(n):
        diag_sum += A3[i][i]
    return diag_sum // 6

'''
********************************************************************************
MAIN
********************************************************************************
'''

''' COMMENTED OUT for GRADESCOPE --> Threshold Checking + Prints Triangles + Expected'''
# def main():
#     compare_strassen_vs_conventional()
    # comb_1024_3 = math.comb(N, 3)
    # results = []

    # # Run 5 separate rand. graphs for each p, then average counted triangles
    # NUM_RUNS = 3
    # for p in PROBS:
    #     # times and triangle counts 
    #     sum_triangles = 0
    #     sum_build_time = 0.0
    #     sum_count_time = 0.0

    #     for _ in range(NUM_RUNS):
    #         # build adjacency
    #         start_build = time.perf_counter()
    #         A = build_random_adjacency(N, p)
    #         end_build = time.perf_counter()

    #         build_time = end_build - start_build
    #         sum_build_time += build_time

    #         # coount triangles
    #         start_count = time.perf_counter()
    #         tri_count = count_triangles(A)
    #         end_count = time.perf_counter()

    #         count_time = end_count - start_count
    #         sum_count_time += count_time

    #         sum_triangles += tri_count

    #     # Average results and print vs expected
    #     avg_triangles = sum_triangles / NUM_RUNS
    #     avg_build = sum_build_time / NUM_RUNS
    #     avg_count = sum_count_time / NUM_RUNS
    #     expected = comb_1024_3 * (p**3)
    #     print(f"\n=== p = {p} ===")
    #     print(f"Average build adjacency time: {avg_build:.2f}s")
    #     print(f"Average triangle count time: {avg_count:.2f}s")
    #     print(f"Average # triangles: {avg_triangles:.1f},  Expected ~ {expected:.1f}")

    #     results.append((p, avg_triangles, expected))

    # # Final table
    # print("\n===== Final Results (averaged over 5 runs) =====")
    # print("p\tTriangles\tExpected")
    # for (p, avg_tri, exp) in results:
    #     print(f"{p}\t{avg_tri:.1f}\t\t{exp:.1f}")


def main():
    if len(sys.argv) != 4:
        print("Usage: {} <flag> <dimension> <inputfile>".format(sys.argv[0]))
        sys.exit(1)

    # parse
    flag = sys.argv[1]
    d = int(sys.argv[2])
    inputfile = sys.argv[3]

    # read file for 2*d*d integers
    with open(inputfile, "r") as f:
        nums = [int(line.strip()) for line in f if line.strip()]
    if len(nums) != 2*d*d:
        print("Error: input file does not have 2*d*d = {} integers.".format(2*d*d))
        sys.exit(1)

    # Converting Flat Matrices A,B
    Anums = nums[:d*d]
    Bnums = nums[d*d:2*d*d]
    A = [Anums[i*d:(i+1)*d] for i in range(d)]
    B = [Bnums[i*d:(i+1)*d] for i in range(d)]

    # Running through GENERAL (for all values of n) strassen multiply
    C = strassen_multiply_gen(A, B)
    # one per line + trailing newline
    for i in range(d):
        print(C[i][i])

if __name__ == "__main__":
    main()