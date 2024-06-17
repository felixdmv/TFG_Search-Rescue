import os
import tkinter as tk
from tkinter import filedialog
import glob
import shutil

def buscaFichero(directorio, nombreFichero, extension):
    nombreImagenSinExtension = nombreFichero.split('.')[0]
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.endswith(extension) and file.startswith(nombreImagenSinExtension):
                return os.path.join(root, file)
    return None

def buscaFicheroMismoNombre(filePath, extension):
    # Obtener el directorio del archivo de imagen
    directorio = os.path.dirname(filePath)
    
    # Obtener el basename del archivo de imagen sin la extensión
    basename = os.path.splitext(os.path.basename(filePath))[0]
    
    # Crear el patrón de búsqueda para archivos XML con el mismo basename
    patronBusqueda = os.path.join(directorio, '**', f'{basename}.' + extension)
    
    # Buscar archivos XML usando glob
    files = glob.glob(patronBusqueda, recursive=True)
    
    if files:
        return files[0]  # Retornar el primer archivo encontrado
    else:
        return None  # Retornar None si no se encuentra ningún archivo
    
def obtieneNombreBase(filePath):
    return os.path.splitext(os.path.basename(filePath))[0]

   
def obtienePathFromBasename(path, basename, extension):
    return os.path.join(path, f"{basename}.{extension}")

def creaPathDirectorioMismoNivel(basePath, directorio):
    return os.path.join(os.path.dirname(basePath), directorio)

def creaPathDirectorioNivelInferior(basePath, subdirectorio):
    return os.path.join(basePath, subdirectorio)

def existePath(path):
    return os.path.exists(path)

def borraDirectorioYContenido(path):
    shutil.rmtree(path)

def creaDirectorio(path, exist_ok=False):
    os.makedirs(path, exist_ok=exist_ok)

def obtieneNombresBase(directorio, extensionesPermitidas=None):
    """ Retorna un conjunto de nombres base de archivos en un directorio """
    baseNames = []
    if extensionesPermitidas is not None:
        extensionesPermitidas = ['.' + extension.lower() for extension in extensionesPermitidas]
    
    for archivo in os.listdir(directorio):
        basename, extension = os.path.splitext(archivo)
        if extensionesPermitidas is None or extension.lower() in extensionesPermitidas:
            baseNames.append(basename)
    return baseNames

def obtienePathFicheros(directorio, extensionesPermitidas=None):
    listaPaths = []
    if extensionesPermitidas is not None:
        extensionesPermitidas = ['.' + extension.lower() for extension in extensionesPermitidas]
    
    for archivo in os.listdir(directorio):
        _, extension = os.path.splitext(archivo)
        if extensionesPermitidas is None or extension.lower() in extensionesPermitidas:
            listaPaths.append(os.path.join(directorio, archivo))
    return listaPaths

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