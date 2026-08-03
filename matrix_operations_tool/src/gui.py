import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
from typing import List, Dict, Any, Tuple

from src.matrix_logic import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    transpose_matrix,
    calculate_determinant,
    calculate_inverse,
    calculate_eigenvalues
)

class MatrixApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Matrix Operations Tool")
        self.root.geometry("960x760")
        self.root.minsize(900, 700)
        
        # Set modern dark-mode colors
        self.bg_color = "#1e1e24"      # Deep dark blue-grey
        self.panel_color = "#282a36"   # Dark Dracula panel
        self.text_color = "#f8f8f2"    # Off-white
        self.accent_color = "#6272a4"  # Muted blue-grey
        self.btn_binary = "#50fa7b"    # Bright green
        self.btn_unary = "#8be9fd"     # Bright cyan
        self.btn_clear = "#ff5555"     # Coral red
        self.btn_active = "#bd93f9"    # Soft purple
        
        self.root.configure(bg=self.bg_color)
        
        # Grid sizes (Rows and Columns)
        self.rows_a = tk.IntVar(value=3)
        self.cols_a = tk.IntVar(value=3)
        self.rows_b = tk.IntVar(value=3)
        self.cols_b = tk.IntVar(value=3)
        
        # Entry widgets matrices
        self.entries_a: List[List[tk.Entry]] = []
        self.entries_b: List[List[tk.Entry]] = []
        
        self.setup_styles()
        self.create_widgets()
        self.rebuild_grid_a()
        self.rebuild_grid_b()

    def setup_styles(self):
        """Sets up ttk widget styles to match the dark theme."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Main frames
        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background=self.panel_color, relief="groove", borderwidth=1)
        
        # Labels
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Helvetica", 10))
        style.configure("Header.TLabel", background=self.bg_color, foreground=self.btn_unary, font=("Helvetica", 14, "bold"))
        style.configure("PanelHeader.TLabel", background=self.panel_color, foreground=self.btn_active, font=("Helvetica", 12, "bold"))
        style.configure("Sub.TLabel", background=self.panel_color, foreground=self.text_color, font=("Helvetica", 10))
        
        # Dropdowns
        style.configure("TCombobox", fieldbackground=self.panel_color, background=self.accent_color, foreground=self.text_color)
        
        # Notebook (Tabs)
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background=self.panel_color, foreground=self.text_color, font=("Helvetica", 10))
        style.map("TNotebook.Tab", background=[("selected", self.accent_color)])

    def create_widgets(self):
        """Builds all user interface layouts."""
        # --- TITLE BANNER ---
        title_label = ttk.Label(self.root, text="🔢 MATRIX OPERATIONS DIAGNOSTICS TOOL", style="Header.TLabel")
        title_label.pack(pady=15)
        
        # --- TOP PANEL: DIMENSION CONTROLS ---
        controls_frame = ttk.Frame(self.root, style="Panel.TFrame")
        controls_frame.pack(fill="x", padx=20, pady=5)
        
        # Matrix A Dimensions
        lbl_dim_a = ttk.Label(controls_frame, text="Matrix A Dimensions:", style="Sub.TLabel")
        lbl_dim_a.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        cb_rows_a = ttk.Combobox(controls_frame, textvariable=self.rows_a, values=[1, 2, 3, 4, 5], width=3, state="readonly")
        cb_rows_a.grid(row=0, column=1, padx=5, pady=10)
        cb_rows_a.bind("<<ComboboxSelected>>", lambda e: self.rebuild_grid_a())
        
        lbl_x_a = ttk.Label(controls_frame, text="x", style="Sub.TLabel")
        lbl_x_a.grid(row=0, column=2, pady=10)
        
        cb_cols_a = ttk.Combobox(controls_frame, textvariable=self.cols_a, values=[1, 2, 3, 4, 5], width=3, state="readonly")
        cb_cols_a.grid(row=0, column=3, padx=5, pady=10)
        cb_cols_a.bind("<<ComboboxSelected>>", lambda e: self.rebuild_grid_a())
        
        # Separator spacing
        ttk.Label(controls_frame, text="   |   ", style="Sub.TLabel").grid(row=0, column=4, padx=15)
        
        # Matrix B Dimensions
        lbl_dim_b = ttk.Label(controls_frame, text="Matrix B Dimensions:", style="Sub.TLabel")
        lbl_dim_b.grid(row=0, column=5, padx=10, pady=10, sticky="w")
        
        cb_rows_b = ttk.Combobox(controls_frame, textvariable=self.rows_b, values=[1, 2, 3, 4, 5], width=3, state="readonly")
        cb_rows_b.grid(row=0, column=6, padx=5, pady=10)
        cb_rows_b.bind("<<ComboboxSelected>>", lambda e: self.rebuild_grid_b())
        
        lbl_x_b = ttk.Label(controls_frame, text="x", style="Sub.TLabel")
        lbl_x_b.grid(row=0, column=7, pady=10)
        
        cb_cols_b = ttk.Combobox(controls_frame, textvariable=self.cols_b, values=[1, 2, 3, 4, 5], width=3, state="readonly")
        cb_cols_b.grid(row=0, column=8, padx=5, pady=10)
        cb_cols_b.bind("<<ComboboxSelected>>", lambda e: self.rebuild_grid_b())

        # --- MID PANEL: VISUAL GRID LAYOUTS ---
        grids_frame = ttk.Frame(self.root)
        grids_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left Panel: Matrix A Grid
        self.frame_a = ttk.Frame(grids_frame, style="Panel.TFrame")
        self.frame_a.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Right Panel: Matrix B Grid
        self.frame_b = ttk.Frame(grids_frame, style="Panel.TFrame")
        self.frame_b.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # --- OPERATIONS & BUTTONS DASHBOARD ---
        ops_frame = ttk.Frame(self.root, style="Panel.TFrame")
        ops_frame.pack(fill="x", padx=20, pady=5)
        
        # Binary Operations
        binary_title = ttk.Label(ops_frame, text="Binary Operations:", style="Sub.TLabel")
        binary_title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        btn_add = tk.Button(ops_frame, text="A + B", command=self.on_add, bg=self.btn_binary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=10)
        btn_add.grid(row=0, column=1, padx=5, pady=10)
        
        btn_sub = tk.Button(ops_frame, text="A - B", command=self.on_subtract, bg=self.btn_binary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=10)
        btn_sub.grid(row=0, column=2, padx=5, pady=10)
        
        btn_mul = tk.Button(ops_frame, text="A × B", command=self.on_multiply, bg=self.btn_binary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=10)
        btn_mul.grid(row=0, column=3, padx=5, pady=10)
        
        # Separator line
        ttk.Label(ops_frame, text="   |   ", style="Sub.TLabel").grid(row=0, column=4, padx=10)
        
        # Unary Operations Choice (Select target matrix)
        self.unary_target = tk.StringVar(value="A")
        lbl_unary = ttk.Label(ops_frame, text="Unary on:", style="Sub.TLabel")
        lbl_unary.grid(row=0, column=5, padx=5, pady=10)
        
        rb_a = tk.Radiobutton(ops_frame, text="A", variable=self.unary_target, value="A", bg=self.panel_color, fg=self.text_color, selectcolor=self.bg_color, activebackground=self.panel_color, activeforeground=self.btn_active)
        rb_a.grid(row=0, column=6, padx=2, pady=10)
        
        rb_b = tk.Radiobutton(ops_frame, text="B", variable=self.unary_target, value="B", bg=self.panel_color, fg=self.text_color, selectcolor=self.bg_color, activebackground=self.panel_color, activeforeground=self.btn_active)
        rb_b.grid(row=0, column=7, padx=2, pady=10)
        
        # Unary Action Buttons
        btn_trans = tk.Button(ops_frame, text="Transpose", command=self.on_transpose, bg=self.btn_unary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=8)
        btn_trans.grid(row=0, column=8, padx=5, pady=10)
        
        btn_det = tk.Button(ops_frame, text="Determinant", command=self.on_determinant, bg=self.btn_unary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=8)
        btn_det.grid(row=0, column=9, padx=5, pady=10)
        
        btn_inv = tk.Button(ops_frame, text="Inverse", command=self.on_inverse, bg=self.btn_unary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=8)
        btn_inv.grid(row=0, column=10, padx=5, pady=10)
        
        btn_eig = tk.Button(ops_frame, text="Eigenvalues", command=self.on_eigenvalues, bg=self.btn_unary, fg="#000000", font=("Helvetica", 10, "bold"), relief="raised", padx=8)
        btn_eig.grid(row=0, column=11, padx=5, pady=10)

        # --- BOTTOM PANEL: RESULTS TEXTBOX ---
        results_frame = ttk.Frame(self.root, style="Panel.TFrame")
        results_frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        
        res_header = ttk.Frame(results_frame, style="Panel.TFrame")
        res_header.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(res_header, text="📝 STRUCTURED RESULTS OUTPUT", style="PanelHeader.TLabel").pack(side="left", pady=5)
        
        btn_clear_res = tk.Button(res_header, text="Clear Output", command=self.clear_output, bg=self.btn_clear, fg="#ffffff", font=("Helvetica", 9, "bold"), relief="flat", bd=0, padx=8)
        btn_clear_res.pack(side="right", pady=5)
        
        # Result textbox
        self.text_output = tk.Text(results_frame, height=8, bg="#121214", fg="#50fa7b", insertbackground="white", font=("Courier New", 11), relief="sunken", bd=1)
        self.text_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_message("System initialized. Enter matrices and select operations.")

    # --- GRID BUILDERS ---
    def rebuild_grid_a(self):
        """Dynamically destroys and recreates Entry boxes for Matrix A."""
        # Clear existing widgets
        for row in self.frame_a.winfo_children():
            row.destroy()
            
        # Panel Header
        header_frame = ttk.Frame(self.frame_a, style="Panel.TFrame")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(header_frame, text="Matrix A", style="PanelHeader.TLabel").pack(side="left")
        
        # Grid layout container
        grid_container = ttk.Frame(self.frame_a, style="Panel.TFrame")
        grid_container.pack(expand=True, pady=10)
        
        r = self.rows_a.get()
        c = self.cols_a.get()
        self.entries_a = []
        
        for i in range(r):
            row_entries = []
            for j in range(c):
                entry = tk.Entry(grid_container, width=8, bg="#343746", fg=self.text_color, insertbackground="white", font=("Helvetica", 11), justify="center", bd=1, relief="solid")
                entry.grid(row=i, column=j, padx=4, pady=4, ipady=4)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.entries_a.append(row_entries)
            
        # Helper action buttons at bottom of frame
        btn_frame = ttk.Frame(self.frame_a, style="Panel.TFrame")
        btn_frame.pack(fill="x", side="bottom", pady=10)
        
        tk.Button(btn_frame, text="Randomize", command=lambda: self.fill_random("A"), bg=self.accent_color, fg=self.text_color, relief="flat", font=("Helvetica", 9), bd=0, padx=6).pack(side="left", padx=15)
        tk.Button(btn_frame, text="Reset to 0", command=lambda: self.clear_grid("A"), bg=self.btn_clear, fg="#ffffff", relief="flat", font=("Helvetica", 9), bd=0, padx=6).pack(side="right", padx=15)

    def rebuild_grid_b(self):
        """Dynamically destroys and recreates Entry boxes for Matrix B."""
        # Clear existing widgets
        for row in self.frame_b.winfo_children():
            row.destroy()
            
        # Panel Header
        header_frame = ttk.Frame(self.frame_b, style="Panel.TFrame")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(header_frame, text="Matrix B", style="PanelHeader.TLabel").pack(side="left")
        
        # Grid layout container
        grid_container = ttk.Frame(self.frame_b, style="Panel.TFrame")
        grid_container.pack(expand=True, pady=10)
        
        r = self.rows_b.get()
        c = self.cols_b.get()
        self.entries_b = []
        
        for i in range(r):
            row_entries = []
            for j in range(c):
                entry = tk.Entry(grid_container, width=8, bg="#343746", fg=self.text_color, insertbackground="white", font=("Helvetica", 11), justify="center", bd=1, relief="solid")
                entry.grid(row=i, column=j, padx=4, pady=4, ipady=4)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.entries_b.append(row_entries)
            
        # Helper action buttons at bottom of frame
        btn_frame = ttk.Frame(self.frame_b, style="Panel.TFrame")
        btn_frame.pack(fill="x", side="bottom", pady=10)
        
        tk.Button(btn_frame, text="Randomize", command=lambda: self.fill_random("B"), bg=self.accent_color, fg=self.text_color, relief="flat", font=("Helvetica", 9), bd=0, padx=6).pack(side="left", padx=15)
        tk.Button(btn_frame, text="Reset to 0", command=lambda: self.clear_grid("B"), bg=self.btn_clear, fg="#ffffff", relief="flat", font=("Helvetica", 9), bd=0, padx=6).pack(side="right", padx=15)

    # --- GRID HELPERS ---
    def fill_random(self, target: str):
        """Populates the selected matrix with random integers between -9 and 9."""
        entries = self.entries_a if target == "A" else self.entries_b
        for row in entries:
            for entry in row:
                entry.delete(0, tk.END)
                entry.insert(0, str(random.randint(-9, 9)))

    def clear_grid(self, target: str):
        """Resets all entries of the selected matrix grid to 0."""
        entries = self.entries_a if target == "A" else self.entries_b
        for row in entries:
            for entry in row:
                entry.delete(0, tk.END)
                entry.insert(0, "0")

    # --- DATA EXTRACTION & MATRICES PARSING ---
    def parse_matrix(self, target: str) -> np.ndarray:
        """Reads values from input Entry widgets and returns a NumPy ndarray."""
        entries = self.entries_a if target == "A" else self.entries_b
        rows = self.rows_a.get() if target == "A" else self.rows_b.get()
        cols = self.cols_a.get() if target == "A" else self.cols_b.get()
        
        matrix_data = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                val_str = entries[i][j].get().strip()
                if not val_str:
                    raise ValueError(f"Empty cell encountered in Matrix {target} at row {i+1}, column {j+1}.")
                try:
                    # Support floating point inputs
                    matrix_data[i, j] = float(val_str)
                except ValueError:
                    raise ValueError(f"Invalid numeric input '{val_str}' in Matrix {target} at row {i+1}, column {j+1}.")
                    
        return matrix_data

    # --- OUTPUT LOGGERS ---
    def log_message(self, message: str):
        """Appends status information or operational descriptions to the log."""
        self.text_output.configure(state="normal")
        self.text_output.insert(tk.END, f"> {message}\n")
        self.text_output.see(tk.END)
        self.text_output.configure(state="disabled")
        
    def log_result(self, header: str, matrix: np.ndarray):
        """Formats and displays a resulting NumPy array cleanly in the log."""
        self.text_output.configure(state="normal")
        self.text_output.insert(tk.END, f"\n[Result] {header}:\n")
        
        # Find maximum length of float representation to align columns
        formatted_rows = []
        for row in matrix:
            formatted_row = [f"{val:,.4f}".rstrip('0').rstrip('.') for val in row]
            # Replace empty strings (e.g. from 0.0) with "0"
            formatted_row = [val if val != "" else "0" for val in formatted_row]
            formatted_rows.append(formatted_row)
            
        col_widths = [max(len(row[col_idx]) for row in formatted_rows) for col_idx in range(matrix.shape[1])]
        
        for row in formatted_rows:
            row_str = "   ".join(f"{val:>{col_widths[idx]}}" for idx, val in enumerate(row))
            self.text_output.insert(tk.END, f"  [ {row_str} ]\n")
            
        self.text_output.insert(tk.END, "-"*50 + "\n")
        self.text_output.see(tk.END)
        self.text_output.configure(state="disabled")

    def log_scalar(self, header: str, value: Union[float, int, complex]):
        """Logs a single numeric value (like determinants or eigenvalues)."""
        self.text_output.configure(state="normal")
        self.text_output.insert(tk.END, f"\n[Result] {header}:\n")
        if isinstance(value, complex):
            # Format complex numbers neatly
            real = f"{value.real:,.4f}".rstrip('0').rstrip('.')
            imag = f"{value.imag:,.4f}".rstrip('0').rstrip('.')
            real = "0" if real == "" else real
            imag = "0" if imag == "" else imag
            
            if value.imag >= 0:
                self.text_output.insert(tk.END, f"  {real} + {imag}i\n")
            else:
                self.text_output.insert(tk.END, f"  {real} - {abs(value.imag):,.4f}i\n")
        else:
            val_str = f"{value:,.6f}".rstrip('0').rstrip('.')
            val_str = "0" if val_str == "" else val_str
            self.text_output.insert(tk.END, f"  {val_str}\n")
            
        self.text_output.insert(tk.END, "-"*50 + "\n")
        self.text_output.see(tk.END)
        self.text_output.configure(state="disabled")

    def clear_output(self):
        """Clears the scrollable textbox and adds a default log."""
        self.text_output.configure(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.configure(state="disabled")
        self.log_message("Structured log cleared.")

    # --- OPERATION EVENTS HANDLERS ---
    def on_add(self):
        try:
            A = self.parse_matrix("A")
            B = self.parse_matrix("B")
            res = add_matrices(A, B)
            self.log_result("Matrix A + Matrix B", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_subtract(self):
        try:
            A = self.parse_matrix("A")
            B = self.parse_matrix("B")
            res = subtract_matrices(A, B)
            self.log_result("Matrix A - Matrix B", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_multiply(self):
        try:
            A = self.parse_matrix("A")
            B = self.parse_matrix("B")
            res = multiply_matrices(A, B)
            self.log_result("Matrix A × Matrix B (Dot Product)", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_transpose(self):
        target = self.unary_target.get()
        try:
            M = self.parse_matrix(target)
            res = transpose_matrix(M)
            self.log_result(f"Transpose of Matrix {target} (M^T)", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_determinant(self):
        target = self.unary_target.get()
        try:
            M = self.parse_matrix(target)
            res = calculate_determinant(M)
            self.log_scalar(f"Determinant of Matrix {target} |M|", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_inverse(self):
        target = self.unary_target.get()
        try:
            M = self.parse_matrix(target)
            res = calculate_inverse(M)
            self.log_result(f"Inverse of Matrix {target} (M^-1)", res)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")

    def on_eigenvalues(self):
        target = self.unary_target.get()
        try:
            M = self.parse_matrix(target)
            vals, vecs = calculate_eigenvalues(M)
            
            self.text_output.configure(state="normal")
            self.text_output.insert(tk.END, f"\n[Result] Eigenvalues of Matrix {target}:\n")
            for idx, val in enumerate(vals):
                # Format eigenvalues which can be complex
                if np.iscomplex(val):
                    self.text_output.insert(tk.END, f"  λ{idx+1} = {val.real:,.4f} + {val.imag:,.4f}i\n")
                else:
                    self.text_output.insert(tk.END, f"  λ{idx+1} = {val.real:,.4f}\n")
            
            self.text_output.insert(tk.END, f"\n[Result] Corresponding Eigenvectors:\n")
            for col_idx in range(vecs.shape[1]):
                col_vec = vecs[:, col_idx]
                col_str = ", ".join(f"{val:,.4f}" if not np.iscomplex(val) else f"{val.real:,.4f}+{val.imag:,.4f}i" for val in col_vec)
                self.text_output.insert(tk.END, f"  v{col_idx+1} = [ {col_str} ]\n")
                
            self.text_output.insert(tk.END, "-"*50 + "\n")
            self.text_output.see(tk.END)
            self.text_output.configure(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.log_message(f"Error: {e}")
