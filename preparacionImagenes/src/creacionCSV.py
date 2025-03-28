import settingsPreparacion as settings
from utils.entradaSalida import cargaParametrosConfiguracionYAML
import utils.utilidadesDirectorios as ud
import utils.dialogoFicheros as dfich
from utils.procesadoXML import getListaBndbox
import csv
from sklearn.model_selection import StratifiedKFold


def createCsv(subimageTypePath, numCajas):
    """
    Create a CSV file with information about the images in the given subimage type path.

    Args:
        subimageTypePath (str): The path to the subimage type directory.
        numCajas (int): The number of boxes to split the subimages into.

    Returns:
        None
    """
    nomFolder = ud.obtieneNombreBase(subimageTypePath)
    nomFich = '_' +  nomFolder + '.csv' 
    csvFilepath = ud.creaPathDirectorioNivelInferior(subimageTypePath, nomFich)  
    
    with open(csvFilepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Nombre del archivo", "Hay humano", "Caja Hay humano"])

        # Lista para almacenar los nombres de archivo de las imágenes JPG
        imageNames = []
        # Lista para almacenar las etiquetas de las imágenes
        labels = []

        listaBasenames = ud.obtieneNombresBase(subimageTypePath, ['jpg'])
        if listaBasenames == []:
            print(f"No se encontraron imágenes en {subimageTypePath}")
            return
        
        for basename in listaBasenames:
            xmlPath = ud.creaPathDirectorioNivelInferior(subimageTypePath, f"{basename}.xml") 
            # Obtener información sobre los objetos en la subimagen
            listaBndbox = getListaBndbox(xmlPath)
            # Determinar la etiqueta de la subimagen
            label = 1 if listaBndbox else 0
            # Agregar el nombre de archivo de la imagen y la etiqueta a las listas
            imageNames.append(basename + '.jpg')
            labels.append(label)

        # Realizar el split de las subimágenes en cajas con StratifiedKFold
        skf = StratifiedKFold(n_splits=numCajas, shuffle=True)

        # Dividir los nombres de archivo y las etiquetas en las cajas
        for i, (_, test_index) in enumerate(skf.split(imageNames, labels)):
            for idx in test_index:
                # Escribir en el CSV el nombre de la subimagen, la etiqueta y el número de caja
                writer.writerow([nomFolder, imageNames[idx], labels[idx], i+1])


def main():
    """
    Main function for creating CSV files based on configuration settings.

    This function loads the configuration parameters from a YAML file,
    selects a directory for subimages, obtains the file paths for the subimages,
    and creates CSV files for each subimage type.

    Returns:
        None
    """
    configuracion = cargaParametrosConfiguracionYAML(settings.PATH_PARAMETROS)
    if configuracion == None:
        print(f"Error cargando el fichero de configuración {settings.PATH_PARAMETROS}")
        return

    print("Creación de CSVs")
    numCajas = configuracion['validacionCruzada']['numCajas']
    subimagesPath = dfich.seleccionaDirectorio()  # Provisional para pruebas
    listasubimageTypePaths = ud.obtienePathFicheros(subimagesPath)
    
    for subimageTypePath in listasubimageTypePaths:
        try:
            createCsv(subimageTypePath, numCajas)
        except Exception as e:
            print(f"Error creando CSV's en {subimageTypePath}: {str(e)}")


if __name__ == '__main__':
    main()