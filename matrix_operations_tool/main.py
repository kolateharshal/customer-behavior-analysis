import tkinter as tk
import sys
import os

# Add the project folder to the Python lookup path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.gui import MatrixApp

def main():
    # Instantiate the standard Tkinter main window
    root = tk.Tk()
    
    # Load our custom Matrix Application
    app = MatrixApp(root)
    
    # Run the window display main loop
    root.mainloop()

if __name__ == "__main__":
    main()
