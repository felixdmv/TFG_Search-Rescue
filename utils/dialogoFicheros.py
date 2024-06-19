import tkinter as tk
from tkinter import filedialog

def seleccionaDirectorio():
   root = tk.Tk()
   root.withdraw()
   root.wm_attributes('-topmost', 1)
   folderPath = filedialog.askdirectory(master=root)
   root.destroy()
   if folderPath == '':
       return None
   return folderPath

def seleccionaFichero():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    filePath = filedialog.askopenfilename(master=root)
    root.destroy()
    if filePath == '':
        return None
    return filePath