import settingsPreparacion as settings
from utils.entradaSalida import cargaParametrosConfiguracionYAML
import utils.utilidadesDirectorios as ud
from utils.procesadoXML import getListaBndbox, createXmlSubimage
from utils.expresionesRegulares import getPatronFile
from utils.graficosImagenes import creaListaRectangulosConIndices
import utils.dialogoFicheros as dfich
from PIL import Image


def bndboxDentro(r1, r2):
    """
    Checks if the bounding box represented by r2 is completely inside the bounding box represented by r1.

    Parameters:
    r1 (tuple): The coordinates of the first bounding box in the format (left, upper, right, lower).
    r2 (tuple): The coordinates of the second bounding box in the format (left, upper, right, lower).

    Returns:
    bool: True if r2 is completely inside r1, False otherwise.
    """
    left1, upper1, right1, lower1 = r1
    left2, upper2, right2, lower2 = r2
    return left2 >= left1 and upper2 >= upper1 and right2 <= right1 and lower2 <= lower1


def bndboxFuera(r1, r2):
    """
    Checks if two bounding boxes are completely outside of each other.

    Args:
        r1 (tuple): The coordinates of the first bounding box in the format (left, upper, right, lower).
        r2 (tuple): The coordinates of the second bounding box in the format (left, upper, right, lower).

    Returns:
        bool: True if the bounding boxes are completely outside of each other, False otherwise.
    """
    left1, upper1, right1, lower1 = r1
    left2, upper2, right2, lower2 = r2
    return right2 < left1 or left2 > right1 or lower2 < upper1 or upper2 > lower1


def situacionBndBox(listaBndbox, rectangulo):
    """
    Determines the position of bounding boxes relative to a given rectangle.

    Args:
        listaBndbox (list): A list of bounding boxes in the format (xmin, ymin, xmax, ymax).
        rectangulo (tuple): A tuple representing the rectangle in the format (left, upper, right, lower).

    Returns:
        tuple: A tuple containing two elements:
            - listaBndboxDentro (list): A list of bounding boxes that are inside the rectangle, with coordinates relative to the rectangle.
            - contadorFuera (int): The count of bounding boxes that are outside the rectangle.

    """
    contadorFuera = 0
    listaBndboxDentro = []
    left, upper, _, _ = rectangulo
    for bndBox in listaBndbox:
        if bndboxDentro(rectangulo, bndBox):
            xmin, ymin, xmax, ymax = bndBox
            # Se crea el bndbox relativo de la subimagen
            new_xmin = xmin - left
            new_ymin = ymin - upper
            new_xmax = xmax - left
            new_ymax = ymax - upper
            listaBndboxDentro.append((new_xmin, new_ymin, new_xmax, new_ymax))
        elif bndboxFuera(rectangulo, bndBox):
            contadorFuera += 1
    return listaBndboxDentro, contadorFuera


def creaSubimagenesYXML(subimagesPath, pathYNames, subimageSize, overlap, margins):
    """
    Creates subimages and XML files based on the given parameters.

    Args:
        subimagesPath (str): The path where the subimages will be saved.
        pathYNames (tuple): A tuple containing the image path, XML path, image name, and image type.
        subimageSize (tuple): A tuple containing the width and height of the subimages.
        overlap (int): The amount of overlap between subimages.
        margins (tuple): A tuple containing the left, top, right, and bottom margins for cropping the subimages.

    Returns:
        tuple: A tuple containing the total number of subimages created, the number of subimages containing humans,
               the number of subimages without humans, and the number of discarded subimages.
    """
    imagePath, xmlPath, imageName, imageType = pathYNames
    print(f"Procesando {imageName}...")
    
    image = Image.open(imagePath)

    subimageTypePath = ud.creaPathDirectorioNivelInferior(subimagesPath, imageType)
    listaBndbox = getListaBndbox(xmlPath)

    listaRectangulosConIndices = creaListaRectangulosConIndices(image.size, subimageSize, overlap, margins)

    numSubimagenesConHumano = 0
    numSubimagenesSinHumano = 0
    numSubimagenesDescartadas = 0
    for rectangulo, indices in listaRectangulosConIndices:
        i, j = indices
        listaBndboxDentro, contadorFuera = situacionBndBox(listaBndbox, rectangulo)
        if len(listaBndboxDentro) > 0 or contadorFuera == len(listaBndbox):
            createXmlSubimage(imageName, subimageTypePath, listaBndboxDentro, i, j)
            subimage = image.crop(rectangulo)
            subimage.save(ud.creaPathDirectorioNivelInferior(subimageTypePath, f"{imageName}_{j}_{i}.jpg"))
        if len(listaBndboxDentro) > 0:
            numSubimagenesConHumano += 1
        elif contadorFuera == len(listaBndbox):
            numSubimagenesSinHumano += 1
        else:
            numSubimagenesDescartadas += 1
    return len(listaRectangulosConIndices), numSubimagenesConHumano, numSubimagenesSinHumano, numSubimagenesDescartadas


def creaPathsYNames(datasetPath, parametros):
    """
    Creates paths and names for images and XML files based on the given dataset path and parameters.

    Args:
        datasetPath (str): The path to the dataset.
        parametros (dict): A dictionary containing the parameters for path creation.

    Returns:
        tuple: A tuple containing the image paths, XML paths, image names, and image types.
    """
    labelPath = datasetPath + '/' + parametros['labelsSubfolder']
    clasificaPorPatron = parametros['clasificaPorPatron']
    expresionRegular = parametros['expresionRegular']
    posicionPatron = parametros['posicionPatron']
    clasificaPorTipoImagen = parametros['directorioUnico']
    imagePaths = ud.obtienePathFicheros(datasetPath, parametros['imageExtensions'])

    xmlPaths = []
    imageNames = []
    tiposImagenes = []
    
    contadorXMLSinPareja = 0
    for imageFile in imagePaths:
        imageName = ud.obtieneNombreBase(imageFile)
        if clasificaPorPatron:
            clasificaPorTipoImagen = getPatronFile(imageName, expresionRegular, posicionPatron)
            if clasificaPorTipoImagen is None:
                print(f"No se encontró un tipo de terreno para la imagen {imageFile}")
                continue
            
        xmlFile = ud.obtienePathFromBasename(labelPath, imageName, 'xml')
        if not ud.existePath(xmlFile):
            contadorXMLSinPareja += 1
            continue
            
        xmlPaths.append(xmlFile)
        imageNames.append(imageName)
        tiposImagenes.append(clasificaPorTipoImagen)

    if contadorXMLSinPareja > 0:
        print(f"Se encontraron {contadorXMLSinPareja} imágenes sin archivo XML correspondiente")
           
    return imagePaths, xmlPaths, imageNames, tiposImagenes


def creacionDirectoriosSubimagenes(datasetPath, nombreFolder, imageTypes):
    """
    Crea directorios para subimágenes en base a un conjunto de tipos de imágenes.

    Args:
        datasetPath (str): Ruta del directorio del dataset.
        nombreFolder (str): Nombre del directorio para las subimágenes.
        imageTypes (list): Lista de tipos de imágenes.

    Returns:
        str: Ruta del directorio de subimágenes creado.
    """
    subimagesPath = ud.creaPathDirectorioMismoNivel(datasetPath, nombreFolder)
    if ud.existePath(subimagesPath):
        print('...Borrando directorio de subimágenes existente...')
        ud.borraDirectorioYContenido(subimagesPath)
        print(f"Se encontró un directorio de subimágenes existente en {subimagesPath}. Se eliminó el directorio.")
    ud.creaDirectorio(subimagesPath, exist_ok=False)
    
    for imageType in set(imageTypes):
        subimageTypeFolder = ud.creaPathDirectorioNivelInferior(subimagesPath, imageType) 
        ud.creaDirectorio(subimageTypeFolder, exist_ok=True)
    return subimagesPath


def generaInformeCreacionSubimagenesXML(ficheroInforme, estadisticas, configuracion):
    """
    Genera un informe de creación de subimágenes y XML.

    Args:
        ficheroInforme (str): La ruta del archivo donde se generará el informe.
        estadisticas (dict): Un diccionario que contiene las estadísticas de las subimágenes.
        configuracion (dict): Un diccionario que contiene la configuración de las subimágenes.

    Returns:
        None
    """
    # Abrir el archivo en modo escritura
    with open(ficheroInforme, 'w', encoding='utf-8') as f:
        f.write("Informe de Creación de Subimágenes y XML\n")
        f.write("=========================================\n\n")

        subimageSize = configuracion['subimages']['size']
        overlap = configuracion['subimages']['overlap']
        margins = configuracion['subimages']['margins']

        f.write(f"Tamaño de subimágenes: {subimageSize}\n")
        f.write(f"Overlap: {overlap}\n")
        f.write(f"Márgenes: {margins}\n\n")

        f.write("=========================================\n\n")
        f.write("Tipo terreno - Num Imágenes - Num Subimágenes - Descartadas - Sin humano - Con humano - % Con humano sobre válidas \n")
        for imageType, datos in estadisticas.items():
            numImagenes = datos['numImagenes']
            numSubimagenes = datos['numSubimagenes']
            numSubimagenesConHumano = datos['numSubimagenesConHumano']
            numSubimagenesSinHumano = datos['numSubimagenesSinHumano']
            numSubimagenesDescartadas = datos['numSubimagenesDescartadas']
            try:
                porcentajeConHumano = numSubimagenesConHumano / (numSubimagenes - numSubimagenesDescartadas) * 100
                f.write(f"{imageType} - {numImagenes} - {numSubimagenes} - {numSubimagenesDescartadas} - {numSubimagenesSinHumano} - {numSubimagenesConHumano} - {porcentajeConHumano:.2f}% \n")
            except ZeroDivisionError:
                f.write(f"No se encontraron subimágenes para {imageType}\n")

        f.write("=========================================\n\n")
        f.write("\nFin del Informe")

    print(f"Informe generado correctamente en '{ficheroInforme}'")


def main():
    """
    Main function for creating subimages and labels.

    This function loads the configuration parameters, selects the directory with the dataset images,
    creates lists of image paths and labels, and then proceeds to create subimages and labels based on the
    specified configuration.

    Returns:
        None
    """
    configuracion = cargaParametrosConfiguracionYAML(settings.PATH_PARAMETROS)
    if configuracion == None:
        print(f"Error cargando el fichero de configuración {settings.PATH_PARAMETROS}")
        return
    
    print("Selección del directorio con las imágenes del dataset")
    datasetPath = dfich.seleccionaDirectorio()
    if datasetPath == None:
        print("No se seleccionó un directorio de imágenes")
        return
    
    print("Creación de listas de paths de imágenes y etiquetas, nombres de imágenes y tipos de imágenes")
    
    parametros = {}
    parametros['labelsSubfolder'] = configuracion['dataSet']['labelsSubfolder']
    parametros['clasificaPorPatron'] = configuracion['dataSetTransformado']['clasificaPorPatron']
    parametros['expresionRegular'] = configuracion['dataSetTransformado']['expresionRegular']
    parametros['posicionPatron'] = configuracion['dataSetTransformado']['posicionPatron']
    parametros['directorioUnico'] = configuracion['dataSetTransformado']['directorioUnico']
    parametros['imageExtensions'] = configuracion['dataSet']['imageExtensions']
    listasPathsYNames = creaPathsYNames(datasetPath, parametros)
    
    print("Creación del directorio de subimágenes y subsubdirectorios por tipos de imágenes")
    subimagesPath = creacionDirectoriosSubimagenes(datasetPath, configuracion['dataSetTransformado']['subimagesFolder'], listasPathsYNames[3])

    print("Creación de subimágenes y etiquetas")

    estadisticas = {}
    for imageType in listasPathsYNames[3]:
        estadisticas[imageType] = {}
        estadisticas[imageType]['numImagenes'] = 0
        estadisticas[imageType]['numSubimagenes'] = 0
        estadisticas[imageType]['numSubimagenesConHumano'] = 0
        estadisticas[imageType]['numSubimagenesSinHumano'] = 0
        estadisticas[imageType]['numSubimagenesDescartadas'] = 0

    for pathsYNames in zip(*listasPathsYNames):  # zip(*) * desempaqueta la lista de listas y zip empareja los elementos de las listas
        try:
            subimageSize = configuracion['subimages']['size']
            overlap = configuracion['subimages']['overlap']
            margins = configuracion['subimages']['margins']
            numSubimagenes, numSubimagenesConHumano, numSubimagenesSinHumano, numSubimagenesDescartadas = creaSubimagenesYXML(subimagesPath, pathsYNames, subimageSize, overlap, margins)
            imageType = pathsYNames[3]
            estadisticas[imageType]['numImagenes'] += 1
            estadisticas[imageType]['numSubimagenes'] += numSubimagenes
            estadisticas[imageType]['numSubimagenesConHumano'] += numSubimagenesConHumano
            estadisticas[imageType]['numSubimagenesSinHumano'] += numSubimagenesSinHumano
            estadisticas[imageType]['numSubimagenesDescartadas'] += numSubimagenesDescartadas
            
        except Exception as e:
            print(f"Error procesando {pathsYNames[2]}: {str(e)}")
            return

    # ficheroInforme = configuracion['informes']['informeCreacionSubimagenes']
    ficheroInforme =  settings.PATH_INFORMECREACIONSUBIMAGENES
    generaInformeCreacionSubimagenesXML(ficheroInforme, estadisticas, configuracion)


if __name__ == '__main__':
    main()