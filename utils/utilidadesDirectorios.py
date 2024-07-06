import os
import glob
import shutil


def obtieneNombreBase(filePath):
    """
    Returns the base name of a file path without the extension.

    Args:
        filePath (str): The file path.

    Returns:
        str: The base name of the file without the extension.
    """
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
    """
    Returns the full path by joining the given path, basename, and extension.

    Args:
        path (str): The base path.
        basename (str): The base name of the file.
        extension (str): The file extension.

    Returns:
        str: The full path formed by joining the path, basename, and extension.
    """
    return os.path.join(path, f"{basename}.{extension}")


def creaPathDirectorioMismoNivel(basePath, directorio):
    """
    Creates a path in the same directory level as the base path.

    Args:
        basePath (str): The base path.
        directorio (str): The name of the directory to be created.

    Returns:
        str: The path of the created directory.

    """
    return os.path.join(os.path.dirname(basePath), directorio)


def creaPathDirectorioNivelInferior(basePath, subdirectorio):
    """
    Creates a path for a subdirectory within a base path.

    Args:
        basePath (str): The base path.
        subdirectorio (str): The name of the subdirectory.

    Returns:
        str: The path to the subdirectory within the base path.
    """
    return os.path.join(basePath, subdirectorio)


def existePath(path):
    """
    Check if a given path exists.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    return os.path.exists(path)


def borraDirectorioYContenido(path):
    """
    Deletes a directory and its contents.

    Args:
        path (str): The path of the directory to be deleted.

    Raises:
        FileNotFoundError: If the directory does not exist.

    """
    shutil.rmtree(path)


def creaDirectorio(path, exist_ok=False):
    """
    Create a directory at the specified path.

    Args:
        path (str): The path where the directory should be created.
        exist_ok (bool, optional): If True, no exception will be raised if the directory already exists. Defaults to False.
    """
    os.makedirs(path, exist_ok=exist_ok)


def obtieneNombresBase(directorio, extensionesPermitidas=None):
    """
    Retorna un conjunto de nombres base de archivos en un directorio.

    Args:
        directorio (str): La ruta del directorio donde se buscarán los archivos.
        extensionesPermitidas (list, optional): Una lista de extensiones permitidas. 
            Si se proporciona, solo se incluirán los archivos con extensiones en esta lista. 
            Por defecto, se incluirán todos los archivos.

    Returns:
        list: Una lista de nombres base de archivos encontrados en el directorio.
    """
    baseNames = []
    if extensionesPermitidas is not None:
        extensionesPermitidas = ['.' + extension.lower() for extension in extensionesPermitidas]
    
    for archivo in os.listdir(directorio):
        basename, extension = os.path.splitext(archivo)
        if extensionesPermitidas is None or extension.lower() in extensionesPermitidas:
            baseNames.append(basename)
    return baseNames


def obtienePathFicheros(directorio, extensionesPermitidas=None):
    """
    Returns a list of file paths in the specified directory that match the given file extensions.

    Args:
        directorio (str): The directory path to search for files.
        extensionesPermitidas (list, optional): A list of file extensions to filter the search. If None, all files will be included. Defaults to None.

    Returns:
        list: A list of file paths that match the specified directory and file extensions.
    """
    listaPaths = []
    if extensionesPermitidas is not None:
        extensionesPermitidas = ['.' + extension.lower() for extension in extensionesPermitidas]
    
    for archivo in os.listdir(directorio):
        _, extension = os.path.splitext(archivo)
        if extensionesPermitidas is None or extension.lower() in extensionesPermitidas:
            listaPaths.append(os.path.join(directorio, archivo))
    return listaPaths