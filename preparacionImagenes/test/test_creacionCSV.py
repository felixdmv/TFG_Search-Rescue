from utils.procesadoXML import getListaBndbox
import csv
import os
from pathlib import Path
from creacionCSV import createCsv


def readCSVToEstructuras(csv_filepath):
    """
    Reads a CSV file and extracts the contents of each column into separate data structures.

    Args:
        csv_filepath (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the following data structures:
            - dataset_set (set): A set containing unique values from the first column of the CSV.
            - filename_list (list): A list containing the values from the second column of the CSV.
            - human_list (list): A list containing the integer values from the third column of the CSV.
            - box_list (list): A list containing the integer values from the fourth column of the CSV.
    """
    # Conjuntos y listas para almacenar los contenidos de las columnas
    dataset_set = set()
    filename_list = []
    human_list = []
    box_list = []
    
    # Leer el archivo CSV
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        # Omitir la cabecera
        next(reader)
        
        # Leer el contenido del archivo CSV
        for row in reader:
            dataset_set.add(row[0])
            filename_list.append(row[1])
            human_list.append(int(row[2]))
            box_list.append(int(row[3]))
    
    return dataset_set, filename_list, human_list, box_list


def obtieneNombresImagenes(directorio, extension='jpg'):
    """
    Obtains the names of images in a given directory with a specified extension.

    Args:
        directorio (str): The directory path where the images are located.
        extension (str, optional): The file extension of the images. Defaults to 'jpg'.

    Returns:
        list: A sorted list of image names with the specified extension in the directory.
    """
    return sorted([f for f in os.listdir(directorio) if f.endswith(f'.{extension}')])
    
    
def test_creacionCSV():
    """
    Test function for the createCsv function.
    
    This function tests the functionality of the createCsv function by performing the following steps:
    1. Sets up the necessary variables and paths.
    2. Creates the directory for the test images if it doesn't exist.
    3. Removes the CSV file if it already exists.
    4. Calls the createCsv function.
    5. Verifies if the CSV file was created in the expected path.
    6. Verifies if the CSV file has the expected headers.
    7. Reads the CSV file and extracts the dataset, image names, human values, and box values.
    8. Verifies if there is only one dataset in the CSV file.
    9. Verifies if the dataset name in the CSV file matches the expected dataset name.
    10. Obtains the image names in the directory.
    11. Verifies if the image names in the CSV file match the image names in the directory.
    12. Verifies the distribution of 0's and 1's in the boxes.
    13. Verifies the "Hay humano" value for each image.
    14. Optionally, cleans up the CSV file after the test.
    """
    # Número arbitrario de cajas para el test
    numCajas = 5
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'PRUEBA_CREAR_CSV'
    # Nombre del archivo CSV que esperamos que se genere
    nomFich = '_PRUEBA_CREAR_CSV.csv'

    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA_CREAR_CSV'
    subimageTypePath = test_dir.joinpath('test_files/' +nomFolder)
    # Ruta donde debería crearse el archivo _PRUEBA_CREAR_CSV.csv
    csvFilepath_expected = subimageTypePath.joinpath(nomFich)
    
    # Asegurarse de que el directorio PRUEBA_CREAR_CSV existe
    subimageTypePath.mkdir(parents=True, exist_ok=True)
    
    # Eliminar el archivo CSV si ya existe
    if csvFilepath_expected.exists():
        os.remove(csvFilepath_expected)

    # Llamar a la función createCsv
    createCsv(subimageTypePath, numCajas)
    
    # Verificar si el archivo CSV se creó en la ruta esperada
    assert csvFilepath_expected.exists(), f"El archivo CSV {nomFich} no se creó en la ruta esperada {csvFilepath_expected}"
    
    # Verificar si el archivo CSV tiene la cabacera adecuada
    with open(csvFilepath_expected, 'r') as file:
        headers = file.readline().strip().split(',')
        assert headers == ["Dataset", "Nombre del archivo", "Hay humano", "Caja Hay humano"], "El archivo CSV no tiene los encabezados esperados"

    dataset_set, image_names_csv, human_list, box_list = readCSVToEstructuras(csvFilepath_expected)
    
    assert len(dataset_set) == 1, "El archivo CSV tiene más de un conjunto de datasets o no tiene filas creadas"
    
    # Verificar que dataset_set coincide con dataset
    assert list(dataset_set)[0] == nomFolder, f"El conjunto de datasets {dataset_set} no coincide con el esperado {nomFolder}"
    
    # Obtener los nombres de las imágenes en el directorio
    image_names_dir = obtieneNombresImagenes(subimageTypePath)

    # Verificar que los nombres de las imágenes en el CSV coinciden con los del directorio
    assert sorted(image_names_csv) == image_names_dir, "Los nombres de las imágenes en el CSV no coinciden con los nombres en el directorio"
  
    # Verificar la distribución de 0's y 1's en las cajas
    caja_dict = {i: {'0s': 0, '1s': 0} for i in range(1, numCajas + 1)}
    for human, caja in zip(human_list, box_list):
        if human == 0:
            caja_dict[caja]['0s'] += 1
        else:
            caja_dict[caja]['1s'] += 1

    listaNumUnos = [caja_dict[caja]['1s'] for caja in caja_dict]
    listaNumCeros = [caja_dict[caja]['0s'] for caja in caja_dict]

    for i in range(len(listaNumUnos)-1):
        assert abs(listaNumUnos[i] - listaNumUnos[i+1]) <= 1, f"La distribución de 1's en las cajas no es equitativa: supera la unidad"
    for i in range(len(listaNumCeros)-1):
        assert abs(listaNumCeros[i] - listaNumCeros[i+1]) <= 1, f"La distribución de 0's en las cajas no es equitativa: supera la unidad"

    # Verificar el valor de "Hay humano" para cada imagen
    for image_name, human_value in zip(image_names_csv, human_list):
        # Obtener la ruta del archivo XML
        xml_path = os.path.join(subimageTypePath, image_name.replace('.jpg', '.xml'))
        
        # Verificar si la imagen tiene humanos
        listaBndbox = getListaBndbox(xml_path)
        tiene_humanos = 1 if listaBndbox else 0
        
        # Verificar si el valor "Hay humano" es correcto
        if tiene_humanos:
            assert human_value == 1, f"La imagen {image_name} tiene humanos, pero el valor 'Hay humano' es {human_value}"
        else:
            assert human_value == 0, f"La imagen {image_name} no tiene humanos, pero el valor 'Hay humano' es {human_value}"

    # Opcional: Limpiar el archivo después del test
    os.remove(csvFilepath_expected)