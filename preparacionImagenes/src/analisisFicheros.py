import settingsPreparacion as settings
from utils.utilidadesDirectorios import obtieneNombresBase
from utils.entradaSalida import cargaParametrosConfiguracionYAML
from utils.dialogoFicheros import seleccionaDirectorio


def generaInformeAnalisisFicheros(ficheroInforme, imagenesSinXML, xmlSinImagen):
    """
    Genera un informe de análisis de ficheros.

    Parameters:
    - ficheroInforme (str): Ruta del archivo donde se generará el informe.
    - imagenesSinXML (list): Lista de nombres de archivos de imagen sin XML asociado.
    - xmlSinImagen (list): Lista de nombres de archivos XML sin imagen asociada.

    Returns:
    None
    """
    # Abrir el archivo en modo escritura
    with open(ficheroInforme, 'w', encoding='utf-8') as f:
        f.write("Informe de Análisis de Ficheros\n")
        f.write("=================================\n\n")
        f.write("Archivos de imagen sin XML asociado:\n")
        f.write(f"Se encontraron {len(imagenesSinXML)} archivos de imagen sin XML asociado.\n\n")
        for nombre_base in imagenesSinXML:
            f.write(f"{nombre_base}\n")
        f.write("\n\nArchivos XML sin imagen asociada:\n")
        f.write(f"Se encontraron {len(xmlSinImagen)} archivos XML sin imagen asociada.\n\n")
        for nombre_base in xmlSinImagen:
            f.write(f"{nombre_base}\n")

        f.write("\nFin del Informe")

    print(f"Informe generado correctamente en '{ficheroInforme}'")


def main():
    """
    Entry point of the program.
    
    This function loads the configuration parameters from a YAML file, selects a dataset directory,
    and performs analysis on the files in the dataset directory and its associated labels directory.
    It generates a report of files that have images without associated XML files, and XML files without
    associated images.
    """
    configuracion = cargaParametrosConfiguracionYAML(settings.PATH_PARAMETROS)
    if configuracion == None:
        print(f"Error cargando el fichero de configuración {settings.PATH_PARAMETROS}")
        return
    
    datasetPath = seleccionaDirectorio()
    if datasetPath == None:
        return
    
    labelsPath = datasetPath + '/' + configuracion['dataSet']['labelsSubfolder']
    
    # Obtener nombres base de archivos de imágenes y XML
    imagesBasenames = set(obtieneNombresBase(datasetPath, configuracion['dataSet']['imageExtensions']))
    xmlBasenames = set(obtieneNombresBase(labelsPath, ['xml']))

    # Encontrar archivos de imagen sin XML asociado
    imagenesSinXML = imagesBasenames - xmlBasenames
    # Encontrar archivos XML sin imagen asociada
    xmlSinImagen = xmlBasenames - imagesBasenames

    ficheroInforme = settings.PATH_INFORMEANALISISFICHEROS
    generaInformeAnalisisFicheros(ficheroInforme, imagenesSinXML, xmlSinImagen)

    
if __name__ == '__main__':
    main()