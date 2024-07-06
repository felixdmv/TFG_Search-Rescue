import json
from PIL import Image
import yaml
import gdown
from defusedxml import ElementTree as ET

def cargaParametrosConfiguracionYAML(ficheroConfiguracion):
    """
    Loads configuration parameters from a YAML file.

    Args:
        ficheroConfiguracion (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.

    """
    try:
        with open(ficheroConfiguracion, 'r', encoding='utf-8') as archivo_config:
            configuracion = yaml.safe_load(archivo_config)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return None
    return configuracion

def cargaArchivoDrive(url, output):
    """
    Downloads a file from the specified URL and saves it to the specified output path.

    Args:
        url (str): The URL of the file to download.
        output (str): The path where the downloaded file should be saved.

    Returns:
        None
    """
    gdown.download(url, output, quiet=False)


def cargaParametrosProcesamiento(ficheroParametros):
    """
    Loads processing parameters from a file.

    Args:
        ficheroParametros (str): The path to the file containing the parameters.

    Returns:
        dict: A dictionary containing the loaded parameters.
    """
    with open(ficheroParametros, 'r') as archivo:
        parametros = json.load(archivo)
    return parametros


def rectangulosEtiquetados(xml_file):
    """
    Extracts labeled rectangles from an XML file.

    Args:
        xml_file (str): The path to the XML file.

    Returns:
        list: A list of tuples representing the labeled rectangles. Each tuple contains the coordinates (xmin, ymin, xmax, ymax) of a rectangle.
    """
    rectangulosConHumano = []
    xml_tree = ET.parse(xml_file)
    root = xml_tree.getroot()
    objects = root.findall('object')

    for obj in objects:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        rectangulosConHumano.append((xmin, ymin, xmax, ymax))

    return rectangulosConHumano

from PIL import Image

def cargaImagen(ficheroImagen):
    """
    Carga una imagen desde un archivo.

    Args:
        ficheroImagen (str): La ruta del archivo de imagen a cargar.

    Returns:
        Image: La imagen cargada.

    """
    imagen = Image.open(ficheroImagen)
    return imagen

if __name__ == '__main__':
    pass