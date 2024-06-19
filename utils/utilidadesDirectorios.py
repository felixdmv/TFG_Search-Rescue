import os

import glob
import shutil

def obtieneNombreBase(filePath):
    return os.path.splitext(os.path.basename(filePath))[0]

def buscaFichero(directorio, nombreFichero, extension):
    """
    Busca un fichero con el nombre y la extensión especificados en el directorio y sus subdirectorios.

    Args:
        directorio (str): El directorio raíz donde se realizará la búsqueda.
        nombreFichero (str): El nombre del fichero a buscar.
        extension (str): La extensión del fichero a buscar.

    Returns:
        str or None: La ruta completa del fichero encontrado, o None si no se encuentra.

    """
    nombreImagenSinExtension = nombreFichero.split('.')[0]
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.endswith(extension) and file.startswith(nombreImagenSinExtension):
                return os.path.join(root, file)
    return None

def buscaFicheroMismoNombreBase(filePath, extension):
    """
    Busca un archivo con el mismo nombre base en el directorio y sus subdirectorios.

    Args:
        filePath (str): La ruta del archivo de referencia.
        extension (str): La extensión del archivo a buscar.

    Returns:
        str or None: La ruta del primer archivo encontrado con el mismo nombre base y extensión especificada,
                     o None si no se encuentra ningún archivo.

    """
    # Obtener el directorio del archivo de imagen
    directorio = os.path.dirname(filePath)
    
    # Obtener el basename del archivo de imagen sin la extensión
    basename = obtieneNombreBase(filePath)
    
    # Crear el patrón de búsqueda para archivos con el mismo basename y extensión
    patronBusqueda = os.path.join(directorio, '**', f'{basename}.' + extension)
    
    # Buscar archivos usando glob
    files = glob.glob(patronBusqueda, recursive=True)
    
    if files:
        return files[0]  # Retornar el primer archivo encontrado
    else:
        return None  # Retornar None si no se encuentra ningún archivo
    


   
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

