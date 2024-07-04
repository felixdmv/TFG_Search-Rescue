from utils.procesadoXML import getListaBndbox
import csv
import os
from pathlib import Path
from src.creacionCSV import createCsv


def readCSVToEstructuras(csv_filepath):
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
    return sorted([f for f in os.listdir(directorio) if f.endswith(f'.{extension}')])
    
# Test para la función createCsv
def test_creacionCSV():
    # Número arbitrario de cajas para el test
    numCajas = 5
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'PRUEBA_TEST_CSV'
    # Nombre del archivo CSV que esperamos que se genere
    nomFich = '_PRUEBA_TEST_CSV.csv'

    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA_TEST_CSV'
    subimageTypePath = test_dir.joinpath('testFiles/' +nomFolder)
    # Ruta donde debería crearse el archivo _PRUEBA_TEST_CSV.csv
    csvFilepath_expected = subimageTypePath.joinpath(nomFich)
    
    
    # Asegurarse de que el directorio PRUEBA_TEST_CSV existe
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
