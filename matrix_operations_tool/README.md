# Matrix Operations Tool - Mathematical Diagnostics & GUI

## 📌 Project Overview
This project is an interactive, desktop-based **Matrix Operations Tool** written in Python. It utilizes the **NumPy** library for high-performance linear algebra calculations and **Tkinter** to provide a clean, dark-themed graphical user interface (GUI).

The application allows users to dynamically size matrices, fill cells manually or with random values, perform binary and unary operations, and view results formatted in a structured output log.

---

## 📂 Project Structure
```
matrix_operations_tool/
├── src/
│   ├── __init__.py
│   ├── matrix_logic.py           # Core NumPy calculations (addition, determinant, eigenvectors, etc.)
│   └── gui.py                    # Tkinter dark-mode desktop window, layout, and event handlers
├── main.py                       # Main application entry point
├── test_logic.py                 # Automated unit tests for linear algebra routines
├── requirements.txt              # NumPy dependency specification
└── README.md                     # Documentation (this file)
```

---

## 📐 Mathematical Operations Reference
The tool performs several linear algebra operations with strict input validation checks:

### 1. Matrix Addition ($A + B$) & Subtraction ($A - B$)
*   **Rule**: The dimensions of Matrix A and Matrix B must match exactly.
*   **Equation**: $C_{i,j} = A_{i,j} \pm B_{i,j}$
*   **Validation**: If shapes differ, the tool raises a dimensional mismatch error.

### 2. Matrix Multiplication ($A \times B$)
*   **Rule**: The number of columns in Matrix A must equal the number of rows in Matrix B.
*   **Equation**: $C_{i,j} = \sum_{k} A_{i,k} \cdot B_{k,j}$
*   **Validation**: Asserts $A.\text{shape}[1] == B.\text{shape}[0]$.

### 3. Transposition ($M^T$)
*   **Rule**: Flips a matrix over its diagonal, switching row indices with column indices.
*   **Equation**: $(M^T)_{i,j} = M_{j,i}$

### 4. Determinant ($|M|$)
*   **Rule**: Calculates the scaling factor of the linear transformation. Defined only for square matrices.
*   **Validation**: Matrix must be square.

### 5. Inverse ($M^{-1}$)
*   **Rule**: Calculates the matrix which, when multiplied by the original, yields the identity matrix.
*   **Equation**: $M \cdot M^{-1} = I$
*   **Validation**: Matrix must be square and non-singular (determinant $\neq 0$). The tool checks if $|M| < 10^{-9}$ and blocks inversion if singular.

### 6. Eigenvalues and Eigenvectors
*   **Rule**: Finds scalar factors $\lambda$ and vectors $v$ such that:
    $$M \cdot v = \lambda \cdot v$$
*   **Validation**: Defined only for square matrices. Handles complex roots cleanly.

---

## 🎨 UI Features & User Guide
The user interface is designed with a premium dark-themed aesthetic (Dracula/macOS style palette) and includes several features to streamline usage:

1.  **Dynamic Input Grids**: Select dimensions from `1x1` to `5x5` using the dropdown selectors. The Entry grids rebuild dynamically while preserving cell layouts.
2.  **Shortcuts**:
    *   **Randomize**: Fills the selected matrix with random integers between `-9` and `9` for rapid testing.
    *   **Reset to 0**: Wipes all entries in the selected matrix to `0`.
3.  **Unified Unary Panel**: Select which matrix ("A" or "B") you want to perform unary operations on, then click Transpose, Determinant, Inverse, or Eigenvalues.
4.  **Structured Log Output**: Results are rendered using a fixed-width font in a console-like window, making rows and columns align. Floats are formatted dynamically to hide trailing zeros.

---

## 🚀 How to Open and Run the Tool

### 1. Install Dependencies
Ensure you have Python 3 and NumPy installed on your computer:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Start the desktop graphical user interface:
```bash
python3 main.py
```

### 3. Run Math Verification Tests
You can run the programmatic mathematical verification suite using:
```bash
python3 test_logic.py
```
This runs the core matrix modules through checks for addition, subtraction, dot product, determinants, inverses, eigenvalues, and exception handling.
