from utils.procesadoXML import getListaBndbox
import csv
import os
from pathlib import Path
from src.creacionSubimagenesXML import creacionDirectoriosSubimagenes, creaPathsYNames, creaSubimagenesYXML
from preparacionImagenes.src.settingsPreparacion import PATH_PARAMETROS
from utils.entradaSalida import cargaParametrosConfiguracionYAML
from utils.utilidadesDirectorios import obtieneNombreBase

    
def test_creaPathsYNames():
    # Nombre del directorio donde se encuentran las imágenes
    nomFolderImagenes = 'testFiles/PRUEBA_TEST_SUBIMAGENESXML'
    nomFolderLabeles = 'testFiles/PRUEBA_TEST_SUBIMAGENESXML/labels'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA'
    datasetPath = test_dir.joinpath(nomFolderImagenes)
    datasetLabelsPath = test_dir.joinpath(nomFolderLabeles)
    
    configuracion = cargaParametrosConfiguracionYAML(PATH_PARAMETROS)
    
    setImageNames = {'train_BRA_1001', 'train_MED_3001', 'train_TRS_0003', 'train_BRK_1003', 'train_ZRI_2004',
                     'train_GOR_1002', 'train_RAK_0003', 'train_CAP_0003', 'train_BLA_0002', 'train_TRS_0002',
                     'train_BRA_1002', 'train_MED_3003', 'train_SB_0001', 'train_BRK_1002', 'train_BRS_0001',
                     'train_VRD_0003', 'train_BRS_0003', 'train_ZRI_2003', 'train_GOR_1001', 'train_BLA_0003',
                     'train_MED_3002', 'train_BRK_1001', 'train_RAK_0002', 'train_BRS_0002', 'train_SB_0003',
                     'train_BRA_1003', 'train_RAK_0001', 'train_GOR_1003', 'train_ZRI_2005', 'train_TRS_0001',
                     'train_CAP_0001', 'train_JAS_0001', 'train_CAP_0002', 'train_SB_0002', 'train_JAS_0002',
                     'train_JAS_0003', 'train_BLA_0001', 'train_VRD_0002', 'train_VRD_0001'}
    

    tiposImagenes_expected = {'BLA', 'BRA', 'BRK', 'BRS', 'CAP', 'GOR',
                              'JAS', 'MED', 'RAK', 'SB', 'TRS', 'VRD', 'ZRI'}
    
    setXML = {setImageNames + '.xml' for setImageNames in setImageNames}
    setJPG = {setImageNames + '.JPG' for setImageNames in setImageNames}
    
    
    # Creo conjunto de Paths esperados. Path() para evitar diferencias en / o \ o // o \\
    xmlPathsSet_expected = {Path(datasetLabelsPath.joinpath(f)) for f in setXML}
    jpgPathsSet_expected = {Path(datasetPath.joinpath(f)) for f in setJPG}

    imagePaths, xmlPaths, imageNames, tiposImagenes = creaPathsYNames(str(datasetPath), configuracion)
    jpgPathSet = {Path(p) for p in imagePaths}
    xmlPathsSet = {Path(p) for p in xmlPaths}

    # Verifica que los paths de las imágenes coinciden con los esperados
    assert jpgPathSet == jpgPathsSet_expected, f"Los paths de las imágenes {jpgPathSet} no coinciden con los esperados {jpgPathsSet_expected}"

    # verifixa que los paths de los XMLs coinciden con los esperados
    assert xmlPathsSet == xmlPathsSet_expected, f"Los paths de los XMLs {xmlPathsSet} no coinciden con los esperados {xmlPathsSet_expected}"

    # Verifica que los nombres de las imágenes coinciden con los esperados
    assert set(imageNames) == setImageNames, f"Los nombres de las imágenes {set(imageNames)} no coinciden con los esperados {setImageNames}"
    
    # Verifica que los tipos de imágenes coinciden con los esperados
    assert set(tiposImagenes) == tiposImagenes_expected, f"Los tipos de imágenes {set(tiposImagenes)} no coinciden con los esperados {tiposImagenes_expected}"


# Test para la función creacionDirectoriosSubimagenes
def test_creacionDirectoriosSubimagenes():
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'testFiles/PRUEBA_TEST_SUBIMAGENESXML'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA'
    datasetPath = test_dir.joinpath(nomFolder)
    
    configuracion = cargaParametrosConfiguracionYAML(PATH_PARAMETROS)
    listasPathsYNames = creaPathsYNames(str(datasetPath), configuracion)

    nombresDirectoriosSubimagen_expected = {'BLA', 'BRA', 'BRK', 'BRS', 'CAP', 'GOR',
                              'JAS', 'MED', 'RAK', 'SB', 'TRS', 'VRD', 'ZRI'}
    subimagesPath = creacionDirectoriosSubimagenes(datasetPath, configuracion['dataSetTransformado']['subimagesFolder'], listasPathsYNames[3])
    nombreDirectorioSubimagenes =  obtieneNombreBase(subimagesPath)
    nombreDirectorioSubimagenes_expected = configuracion['dataSetTransformado']['subimagesFolder']
    # El directorio creado debe llamarse como en configuracion['dataSetTransformado']['subimagesFolder']
    assert nombreDirectorioSubimagenes == nombreDirectorioSubimagenes_expected, f"El directorio creado {nombreDirectorioSubimagenes} no coincide con el esperado 'subimages'" 
    nombresDirectoriosSubimagen = set([obtieneNombreBase(d) for d in Path(subimagesPath).iterdir()])

    # Verifica que se han creado los directorios esperados
    assert nombresDirectoriosSubimagen == nombresDirectoriosSubimagen_expected, f"Los directorios creados {nombresDirectoriosSubimagen} no coinciden con los esperados {nombresDirectoriosSubimagen_expected}"
    

def test_creacreaSubimagenesYXML():
    
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'testFiles/PRUEBA_TEST_SUBIMAGENESXML'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA'
    datasetPath = test_dir.joinpath(nomFolder)
    
    configuracion = cargaParametrosConfiguracionYAML(PATH_PARAMETROS)
    listasPathsYNames = creaPathsYNames(str(datasetPath), configuracion)

    nombresDirectoriosSubimagen_expected = {'BLA', 'BRA', 'BRK', 'BRS', 'CAP', 'GOR',
                              'JAS', 'MED', 'RAK', 'SB', 'TRS', 'VRD', 'ZRI'}
    subimagesPath = creacionDirectoriosSubimagenes(datasetPath, configuracion['dataSetTransformado']['subimagesFolder'], listasPathsYNames[3])
 
    pathImagen = datasetPath.joinpath('train_BLA_0001.JPG')
    pathXML = datasetPath.joinpath('labels/train_BLA_0001.xml')
    nombreImagen = 'train_BLA_0001'
    tipoImagen = 'BLA'
    
    
    
      
    numSubimagenes, numSubimagenesConHumano, numSubimagenesSinHumano, numSubimagenesDescartadas = creaSubimagenesYXML(subimagesPath, (pathImagen, pathXML, nombreImagen, tipoImagen), configuracion)
                                                                                                                                    