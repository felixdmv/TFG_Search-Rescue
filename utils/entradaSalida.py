
import os
import json
import xml.etree.ElementTree as ET
import gdown
from PIL import Image
import yaml

def cargaParametrosConfiguracionYAML(ficheroConfiguracion):
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
    with open(ficheroParametros, 'r') as archivo:
        parametros = json.load(archivo)
    return parametros





# Función para obtener información sobre los objetos en una imagen
def rectangulosEtiquetados(xml_file):
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

def cargaImagen(ficheroImagen):
    imagen = Image.open(ficheroImagen)
    return imagen



if __name__ == '__main__':
    pass
