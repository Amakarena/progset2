#!/usr/bin/env python3
import random
from strassen import (
    conventional_multiply,
    strassen_multiply,
    strassen_multiply_gen
)
'''
********************************************************************************
TEST CONVENTIONAL MULTIPLY
********************************************************************************
'''

def test_conventional_multiply():
    # Test Case 1: 2x2 matrices multiplication.
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    result = conventional_multiply(A, B, 2)
    assert result == expected, f"Test Case 1 Failed: expected {expected}, got {result}"

    # Test Case 2: Multiplying with the identity matrix (2x2).
    I = [[1, 0], [0, 1]]
    C = [[9, 8], [7, 6]]
    result = conventional_multiply(I, C, 2)
    assert result == C, f"Test Case 2 Failed: expected {C}, got {result}"

    # Test Case 3: 3x3 matrices multiplication.
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    expected = [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
    result = conventional_multiply(A, B, 3)
    assert result == expected, f"Test Case 3 Failed: expected {expected}, got {result}"

    print("All conventional tests passed!")

"""
********************************************************************************
HELPER: Generate random matrix of dimension n with entries from `values`.
********************************************************************************
"""
def generate_random_matrix(n, values):
    return [[random.choice(values) for _ in range(n)] for _ in range(n)]

'''
********************************************************************************
TEST BASIC STRASSEN (POWER-OF-2 ONLY)
********************************************************************************
'''
def test_strassen_2x2():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = conventional_multiply(A, B, 2)
    result = strassen_multiply(A, B)
    assert result == expected, f"Strassen 2x2: expected {expected}, got {result}"

def test_strassen_random_power_of_2():
    """Test random power-of-2 sizes for your strassen_multiply."""
    sets = {
        "Binary": [0, 1],
        "Ternary": [0, 1, 2],
        "Signed Ternary": [-1, 0, 1]
    }
    sizes = [2, 4, 8]
    for label, vals in sets.items():
        for n in sizes:
            A = generate_random_matrix(n, vals)
            B = generate_random_matrix(n, vals)
            expected = conventional_multiply(A, B, n)
            result = strassen_multiply(A, B)
            assert result == expected, (
                f"{label} Strassen {n}x{n} failed!\n"
                f"Expected: {expected}\nGot: {result}"
            )
            print(f"{label} Strassen Random {n}x{n} passed.")

def test_strassen_basic():
    """Run all basic Strassen tests (assuming it's power-of-2 oriented)."""
    test_strassen_2x2()
    print("Strassen 2x2 test passed.")
    test_strassen_random_power_of_2()
    print("All basic Strassen (power-of-2) tests passed!\n")

"""
********************************************************************************
TEST GENERALIZED STRASSEN (PADDING)
********************************************************************************
"""
from strassen import strassen_multiply_gen

def test_gen_2x2():
    """Even though 2x2 is already a power of 2, let's confirm strassen_multiply_gen is identical."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = conventional_multiply(A, B, 2)
    result = strassen_multiply_gen(A, B)
    assert result == expected, f"Gen Strassen 2x2: expected {expected}, got {result}"

def test_gen_3x3():
    """Specifically test a 3x3 with the general version, which must pad to 4x4."""
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    expected = conventional_multiply(A, B, 3)
    result = strassen_multiply_gen(A, B)
    assert result == expected, f"Gen Strassen 3x3: expected {expected}, got {result}"

def test_gen_identity_5x5():
    """5x5 identity test => must pad to 8x8 internally."""
    I = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
    A = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    expected = A
    result = strassen_multiply_gen(I, A)
    assert result == expected, f"Gen Strassen 5x5 Identity: expected {expected}, got {result}"

def test_gen_random_non_power_sizes():
    """Try random NxN for non-powers of 2: e.g. 3, 5, 6, 7, 9, etc."""
    sets = {
        "Binary": [0, 1],
        "Ternary": [0, 1, 2],
        "Signed Ternary": [-1, 0, 1]
    }
    # Some sample non-power-of-2 sizes
    sizes = [3, 5, 6, 7, 9]
    for label, vals in sets.items():
        for n in sizes:
            A = generate_random_matrix(n, vals)
            B = generate_random_matrix(n, vals)
            expected = conventional_multiply(A, B, n)
            result = strassen_multiply_gen(A, B)
            assert result == expected, (
                f"{label} Gen Strassen {n}x{n} failed!\n"
                f"Expected: {expected}\nGot: {result}"
            )
            print(f"{label} Gen Strassen Random {n}x{n} passed.")

def test_strassen_gen():
    """Runs all the generalized Strassen tests, including non-power-of-2."""
    test_gen_2x2()
    print("Gen Strassen 2x2 test passed.")
    test_gen_3x3()
    print("Gen Strassen 3x3 test passed.")
    test_gen_identity_5x5()
    print("Gen Strassen 5x5 identity test passed.")
    test_gen_random_non_power_sizes()
    print("All Gen Strassen tests (with padding) passed!\n")
    
"""
********************************************************************************
MAIN
********************************************************************************
"""
def main():
    # 1) Test your conventional multiply
    test_conventional_multiply()

    # 2) Test your original Strassen (ideal for power-of-2)
    test_strassen_basic()

    # 3) Test your new generalized Strassen with padding
    test_strassen_gen()

    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
