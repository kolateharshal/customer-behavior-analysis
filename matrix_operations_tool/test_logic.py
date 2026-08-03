import numpy as np
import sys
import os

# Set search path to project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.matrix_logic import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    transpose_matrix,
    calculate_determinant,
    calculate_inverse,
    calculate_eigenvalues
)

def run_tests():
    print("=" * 60)
    print("RUNNING PROGRAMMATIC MATRIX MATH ALGORITHM VERIFICATION")
    print("=" * 60)
    
    # Test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    # 1. Addition
    res_add = add_matrices(A, B)
    expected_add = np.array([[6.0, 8.0], [10.0, 12.0]])
    np.testing.assert_allclose(res_add, expected_add)
    print("✅ Addition algorithm: PASSED")
    
    # 2. Subtraction
    res_sub = subtract_matrices(A, B)
    expected_sub = np.array([[-4.0, -4.0], [-4.0, -4.0]])
    np.testing.assert_allclose(res_sub, expected_sub)
    print("✅ Subtraction algorithm: PASSED")
    
    # 3. Multiplication (Dot Product)
    res_mul = multiply_matrices(A, B)
    expected_mul = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(res_mul, expected_mul)
    print("✅ Multiplication algorithm: PASSED")
    
    # 4. Transpose
    res_trans = transpose_matrix(A)
    expected_trans = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(res_trans, expected_trans)
    print("✅ Transpose algorithm: PASSED")
    
    # 5. Determinant
    res_det = calculate_determinant(A)
    expected_det = -2.0  # 1*4 - 2*3 = -2
    assert abs(res_det - expected_det) < 1e-9
    print("✅ Determinant algorithm: PASSED")
    
    # 6. Inverse
    res_inv = calculate_inverse(A)
    expected_inv = np.array([[-2.0, 1.0], [1.5, -0.5]])
    np.testing.assert_allclose(res_inv, expected_inv)
    print("✅ Inverse algorithm: PASSED")
    
    # 7. Eigenvalues
    vals, vecs = calculate_eigenvalues(A)
    # Check definition: A * v = lambda * v
    for i in range(len(vals)):
        v = vecs[:, i]
        lambd = vals[i]
        np.testing.assert_allclose(np.dot(A, v), lambd * v, atol=1e-9)
    print("✅ Eigenvalues & Eigenvectors math definition: PASSED")
    
    # 8. Check dimension mismatch errors
    C = np.array([[1.0, 2.0, 3.0]]) # 1x3 matrix
    try:
        add_matrices(A, C)
        assert False, "Should raise ValueError for addition dimension mismatch"
    except ValueError:
        print("✅ Addition dimension validation check: PASSED")
        
    try:
        multiply_matrices(A, C)
        assert False, "Should raise ValueError for multiplication dimension mismatch"
    except ValueError:
        print("✅ Multiplication columns-to-rows validation check: PASSED")
        
    try:
        calculate_determinant(C)
        assert False, "Should raise ValueError for determinant non-square matrix"
    except ValueError:
        print("✅ Determinant squareness validation check: PASSED")
        
    D = np.array([[1.0, 2.0], [2.0, 4.0]]) # singular matrix det = 0
    try:
        calculate_inverse(D)
        assert False, "Should raise ValueError for singular matrix inversion"
    except ValueError:
        print("✅ Singular matrix inversion validation check: PASSED")
        
    print("=" * 60)
    print("ALL CORE MATHEMATICAL ALGORITHMS PASSED VERIFICATION!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
