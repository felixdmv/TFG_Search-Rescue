import settingsPreparacion as settings
from utils.utilidadesDirectorios import obtienePathFicheros
from utils.dialogoFicheros import seleccionaDirectorio
from utils.procesadoXML import getListaBndbox
from utils.entradaSalida import cargaParametrosConfiguracionYAML

import sys


def analizaBndbox(xmlPaths):
    """
    Analyzes the bounding boxes in the given XML files and calculates various statistics.

    Args:
        xmlPaths (list): A list of file paths to XML files containing bounding box information.

    Returns:
        tuple: A tuple containing the following statistics:
            - contadorBndbox (int): The total number of bounding boxes analyzed.
            - maxAncho (int): The maximum width of a bounding box.
            - maxAlto (int): The maximum height of a bounding box.
            - minAncho (int): The minimum width of a bounding box.
            - minAlto (int): The minimum height of a bounding box.
            - mediaAncho (float): The average width of a bounding box.
            - mediaAlto (float): The average height of a bounding box.
    """
    contadorBndbox = 0
    maxAncho = 0
    maxAlto = 0
    minAncho = sys.maxsize
    minAlto = sys.maxsize
    sumAncho = 0
    sumAlto = 0
    for xmlPath in xmlPaths:
        listaBdnbox = getListaBndbox(xmlPath)
        for bndBox in listaBdnbox:
            contadorBndbox += 1
            ancho = bndBox[2] - bndBox[0]
            alto = bndBox[3] - bndBox[1]
            sumAncho += ancho
            sumAlto += alto
            if ancho > maxAncho:
                maxAncho = ancho
            if alto > maxAlto:  
                maxAlto = alto
            if ancho < minAncho:
                minAncho = ancho
            if alto < minAlto:
                minAlto = alto
    
    mediaAncho = sumAncho / contadorBndbox
    mediaAlto = sumAlto / contadorBndbox

    return contadorBndbox, maxAncho, maxAlto, minAncho, minAlto, mediaAncho, mediaAlto


def generaInformeBndbox(ficheroInforme, salidaAnalisis):
    """
    Genera un informe de medidas de Bounding Boxes.

    Args:
        ficheroInforme (str): Ruta del archivo donde se generará el informe.
        salidaAnalisis (tuple): Tupla con los valores de salida del análisis de las Bounding Boxes.
            Contiene los siguientes elementos:
            - contadorBndbox (int): Número total de Bounding Boxes.
            - maxAncho (int): Ancho máximo de las Bounding Boxes.
            - maxAlto (int): Alto máximo de las Bounding Boxes.
            - minAncho (int): Ancho mínimo de las Bounding Boxes.
            - minAlto (int): Alto mínimo de las Bounding Boxes.
            - mediaAncho (float): Ancho medio de las Bounding Boxes.
            - mediaAlto (float): Alto medio de las Bounding Boxes.

    Returns:
        None
    """
    contadorBndbox, maxAncho, maxAlto, minAncho, minAlto, mediaAncho, mediaAlto = salidaAnalisis
    # Abrir el archivo en modo escritura
    with open(ficheroInforme, 'w', encoding='utf-8') as f:
        f.write("Informe de Medidas de Bounding Boxes\n")
        f.write("===============================\n\n")
        
        f.write(f"Número total de Bounding Boxes: {contadorBndbox}\n\n")
        
        f.write("Medidas:\n")
        f.write(f"- Ancho máximo: {maxAncho}\n")
        f.write(f"- Alto máximo: {maxAlto}\n")
        f.write(f"- Ancho mínimo: {minAncho}\n")
        f.write(f"- Alto mínimo: {minAlto}\n")
        f.write(f"- Ancho medio: {mediaAncho}\n")
        f.write(f"- Alto medio: {mediaAlto}\n\n")
        
        f.write("\nFin del Informe")

    print(f"Informe generado correctamente en '{ficheroInforme}'")


def main():
    """
    Entry point of the program.
    
    This function loads the configuration parameters, selects a directory, obtains XML file paths,
    performs bounding box analysis, and generates a report.
    """
    configuracion = cargaParametrosConfiguracionYAML(settings.PATH_PARAMETROS)
    if configuracion == None:
        print(f"Error cargando el fichero de configuración {settings.PATH_PARAMETROS}")
        return
    
    labelsPath = seleccionaDirectorio()
    if labelsPath == None:
        return
    
    xmlPaths = obtienePathFicheros(labelsPath, extensionesPermitidas=['xml'])
    salidaAnalisis = analizaBndbox(xmlPaths)

    ficheroInforme = settings.PATH_INFORMEBND
    generaInformeBndbox(ficheroInforme, salidaAnalisis)


if __name__ == '__main__':
    main()