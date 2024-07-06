import tkinter as tk
from tkinter import filedialog

def seleccionaDirectorio():
    """
    Opens a dialog box to select a directory and returns the selected directory path.

    Returns:
        str or None: The selected directory path. Returns None if no directory is selected.
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folderPath = filedialog.askdirectory(master=root)
    root.destroy()
    if folderPath == '':
        return None
    return folderPath

def seleccionaFichero():
    """
    Opens a file dialog to allow the user to select a file.

    Returns:
        str or None: The path of the selected file, or None if no file was selected.
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    filePath = filedialog.askopenfilename(master=root)
    root.destroy()
    if filePath == '':
        return None
    return filePath