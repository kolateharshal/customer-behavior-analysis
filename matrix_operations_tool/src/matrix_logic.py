import numpy as np
from typing import Tuple, Union

def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Adds two matrices A and B."""
    if A.shape != B.shape:
        raise ValueError(f"Dimension mismatch for addition: Matrix A is {A.shape[0]}x{A.shape[1]} and Matrix B is {B.shape[0]}x{B.shape[1]}. Dimensions must match exactly.")
    return A + B


def subtract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Subtracts matrix B from matrix A."""
    if A.shape != B.shape:
        raise ValueError(f"Dimension mismatch for subtraction: Matrix A is {A.shape[0]}x{A.shape[1]} and Matrix B is {B.shape[0]}x{B.shape[1]}. Dimensions must match exactly.")
    return A - B


def multiply_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiplies two matrices A and B (dot product)."""
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimension mismatch for multiplication: Matrix A has {A.shape[1]} columns, but Matrix B has {B.shape[0]} rows. Columns of A must match rows of B.")
    return np.dot(A, B)


def transpose_matrix(A: np.ndarray) -> np.ndarray:
    """Transposes matrix A."""
    return A.T


def calculate_determinant(A: np.ndarray) -> float:
    """Calculates the determinant of a square matrix A."""
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Determinant error: Matrix is not square. Size is {A.shape[0]}x{A.shape[1]}. Matrix must be square (e.g., 2x2, 3x3).")
    det = np.linalg.det(A)
    # Return rounded or float
    return float(det)


def calculate_inverse(A: np.ndarray) -> np.ndarray:
    """Calculates the inverse of a square, non-singular matrix A."""
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Inverse error: Matrix is not square. Size is {A.shape[0]}x{A.shape[1]}. Only square matrices can be inverted.")
    
    det = calculate_determinant(A)
    if abs(det) < 1e-9:
        raise ValueError(f"Inverse error: Matrix is singular (determinant is approximately zero: {det:.2e}). It does not have an inverse.")
        
    return np.linalg.inv(A)


def calculate_eigenvalues(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates eigenvalues and eigenvectors of matrix A.
    Returns:
        eigenvalues: 1D array of eigenvalues (might be complex)
        eigenvectors: 2D array of eigenvectors
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Eigenvalue error: Matrix is not square. Size is {A.shape[0]}x{A.shape[1]}. Eigenvalues are only defined for square matrices.")
    
    vals, vecs = np.linalg.eig(A)
    return vals, vecs
