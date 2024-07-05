import pytest
from pathlib import Path
from src.creacionSubimagenesXML import creacionDirectoriosSubimagenes, creaPathsYNames, creaSubimagenesYXML
from src.settingsPreparacion import PATH_PARAMETROS
from utils.entradaSalida import cargaParametrosConfiguracionYAML
from utils.utilidadesDirectorios import obtieneNombreBase, borraDirectorioYContenido

@pytest.fixture
def parametros():
    configuracion = cargaParametrosConfiguracionYAML(PATH_PARAMETROS)
    parametros = {}
    parametros['labelsSubfolder'] = configuracion['dataSet']['labelsSubfolder']
    parametros['clasificaPorPatron'] = configuracion['dataSetTransformado']['clasificaPorPatron']
    parametros['expresionRegular'] = configuracion['dataSetTransformado']['expresionRegular']
    parametros['posicionPatron'] = configuracion['dataSetTransformado']['posicionPatron']
    parametros['directorioUnico'] = configuracion['dataSetTransformado']['directorioUnico']
    parametros['imageExtensions'] = configuracion['dataSet']['imageExtensions']
    return parametros

@pytest.fixture
def nombres():
    nombres = {}
    nombres['setImageNames'] = {'train_BRA_1001', 'train_MED_3001', 'train_TRS_0003', 'train_BRK_1003', 'train_ZRI_2004',
                     'train_GOR_1002', 'train_RAK_0003', 'train_CAP_0003', 'train_BLA_0002', 'train_TRS_0002',
                     'train_BRA_1002', 'train_MED_3003', 'train_SB_0001', 'train_BRK_1002', 'train_BRS_0001',
                     'train_VRD_0003', 'train_BRS_0003', 'train_ZRI_2003', 'train_GOR_1001', 'train_BLA_0003',
                     'train_MED_3002', 'train_BRK_1001', 'train_RAK_0002', 'train_BRS_0002', 'train_SB_0003',
                     'train_BRA_1003', 'train_RAK_0001', 'train_GOR_1003', 'train_ZRI_2005', 'train_TRS_0001',
                     'train_CAP_0001', 'train_JAS_0001', 'train_CAP_0002', 'train_SB_0002', 'train_JAS_0002',
                     'train_JAS_0003', 'train_BLA_0001', 'train_VRD_0002', 'train_VRD_0001'}
    nombres['tiposImagenes'] = {'BLA', 'BRA', 'BRK', 'BRS', 'CAP', 'GOR',
                              'JAS', 'MED', 'RAK', 'SB', 'TRS', 'VRD', 'ZRI'}
    
    nombres['archivosJPG'] = {'train_BLA_0001_0_0.jpg', 'train_BLA_0001_0_1.jpg', 'train_BLA_0001_0_2.jpg',
                           'train_BLA_0001_0_3.jpg', 'train_BLA_0001_1_0.jpg', 'train_BLA_0001_1_1.jpg',
                           'train_BLA_0001_1_2.jpg', 'train_BLA_0001_1_3.jpg', 'train_BLA_0001_2_0.jpg',
                           'train_BLA_0001_2_1.jpg', 'train_BLA_0001_2_2.jpg', 'train_BLA_0001_2_3.jpg',
                           'train_BLA_0001_3_0.jpg', 'train_BLA_0001_3_1.jpg', 'train_BLA_0001_3_2.jpg',
                           'train_BLA_0001_3_3.jpg'}
    nombres['archivosXML'] = {'train_BLA_0001_0_0.xml', 'train_BLA_0001_0_1.xml', 'train_BLA_0001_0_2.xml',
                            'train_BLA_0001_0_3.xml', 'train_BLA_0001_1_0.xml', 'train_BLA_0001_1_1.xml',
                            'train_BLA_0001_1_2.xml', 'train_BLA_0001_1_3.xml', 'train_BLA_0001_2_0.xml',
                            'train_BLA_0001_2_1.xml', 'train_BLA_0001_2_2.xml', 'train_BLA_0001_2_3.xml',
                            'train_BLA_0001_3_0.xml', 'train_BLA_0001_3_1.xml', 'train_BLA_0001_3_2.xml',
                            'train_BLA_0001_3_3.xml'}
    return nombres
    
def test_creaPathsYNames(parametros, nombres):
    # Nombre del directorio donde se encuentran las imágenes
    nomFolderImagenes = 'test_files/PRUEBA_CREAR_SUBIMAGENESXML'
    nomFolderLabeles = 'test_files/PRUEBA_CREAR_SUBIMAGENESXML/labels'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA'
    datasetPath = test_dir.joinpath(nomFolderImagenes)
    datasetLabelsPath = test_dir.joinpath(nomFolderLabeles)
    
    setImageNames = nombres['setImageNames']

    tiposImagenes_expected = nombres['tiposImagenes']
    
    setXML = {setImageNames + '.xml' for setImageNames in setImageNames}
    setJPG = {setImageNames + '.JPG' for setImageNames in setImageNames}
    
    
    # Creo conjunto de Paths esperados. Path() para evitar diferencias en / o \ o // o \\
    xmlPathsSet_expected = {Path(datasetLabelsPath.joinpath(f)) for f in setXML}
    jpgPathsSet_expected = {Path(datasetPath.joinpath(f)) for f in setJPG}

    imagePaths, xmlPaths, imageNames, tiposImagenes = creaPathsYNames(str(datasetPath), parametros)
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
def test_creacionDirectoriosSubimagenes(parametros, nombres):
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'test_files/PRUEBA_CREAR_SUBIMAGENESXML'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA_CREAR_SUBIMAGENESXML'
    datasetPath = test_dir.joinpath(nomFolder)

    listasPathsYNames = creaPathsYNames(str(datasetPath), parametros)

    nombresDirectoriosSubimagen_expected = nombres['tiposImagenes']
    subimagesPath = creacionDirectoriosSubimagenes(datasetPath, 'subimages', listasPathsYNames[3])
    nombreDirectorioSubimagenes =  obtieneNombreBase(subimagesPath)
    nombreDirectorioSubimagenes_expected = 'subimages'
    # El directorio creado debe llamarse 'subimages'
    assert nombreDirectorioSubimagenes == nombreDirectorioSubimagenes_expected, f"El directorio creado {nombreDirectorioSubimagenes} no coincide con el esperado 'subimages'" 
    nombresDirectoriosSubimagen = set([obtieneNombreBase(d) for d in Path(subimagesPath).iterdir()])

    # Verifica que se han creado los directorios esperados
    assert nombresDirectoriosSubimagen == nombresDirectoriosSubimagen_expected, f"Los directorios creados {nombresDirectoriosSubimagen} no coinciden con los esperados {nombresDirectoriosSubimagen_expected}"
    # Borrar directorio de subimágenes
    borraDirectorioYContenido(subimagesPath)

def test_creaSubimagenesYXML(parametros, nombres):
    # Nombre del directorio donde se encuentran las imágenes
    nomFolder = 'test_files/PRUEBA_CREAR_SUBIMAGENESXML'
    
    # Obtener el directorio padre del archivo de prueba (donde se encuentra el directorio 'test')
    test_dir = Path(__file__).resolve().parents[0]
    # Construir la ruta completa al directorio 'PRUEBA_CREAR_SUBIMAGENESXML'
    datasetPath = test_dir.joinpath(nomFolder)
    
    listasPathsYNames = creaPathsYNames(str(datasetPath), parametros)

    subimagesPath = creacionDirectoriosSubimagenes(datasetPath, 'subimagesFolder', listasPathsYNames[3])
 
    pathImagen = datasetPath.joinpath('train_BLA_0001.JPG')
    pathXML = datasetPath.joinpath('labels/train_BLA_0001.xml')
    nombreImagen = 'train_BLA_0001'
    tipoImagen = 'BLA'
    
    ancho = 40
    alto = 30

    anchuraSubimagen = 20
    alturaSubimagen = 15
    solapamientoAncho = 11
    solapamientoAlto = 8
    margenAncho = 1
    margenAlto = 1

    subimageSize = anchuraSubimagen,  alturaSubimagen
    overlap = solapamientoAncho, solapamientoAlto
    margins = margenAncho, margenAlto
    numSubimagenesAncho =  4
    numSubimagenesAlto = 4

    numSubimagenes_expected = numSubimagenesAncho * numSubimagenesAlto

    numSubimagenes, _,_,_ = creaSubimagenesYXML(subimagesPath, (pathImagen, pathXML, nombreImagen, tipoImagen), subimageSize, overlap, margins)

    assert numSubimagenes == numSubimagenes_expected, f"El número de subimágenes {numSubimagenes} no coincide con el esperado 16"
   
    pathBLA = Path(subimagesPath).joinpath('BLA')
    archivosJPG_expected = nombres['archivosJPG']
    archivosXML_expected = nombres['archivosXML']
    
    # Set de los nombres de los archivos .jpg
    archivos_jpg = {archivo.name for archivo in pathBLA.iterdir() if archivo.is_file() and archivo.suffix == '.jpg'}

    # Set de los nombres de los archivos .xml
    archivos_xml = {archivo.name for archivo in pathBLA.iterdir() if archivo.is_file() and archivo.suffix == '.xml'}

    # Verifica que los archivos .jpg creados coinciden con los esperados
    assert archivos_jpg == archivosJPG_expected, f"Los archivos .jpg creados {archivos_jpg} no coinciden con los esperados {archivosJPG_expected}"
    # Verifica que los archivos .xml creados coinciden con los esperados
    assert archivos_xml == archivosXML_expected, f"Los archivos .xml creados {archivos_xml} no coinciden con los esperados {archivosXML_expected}"

    # Borrar directorio de subimágenes
    borraDirectorioYContenido(subimagesPath)



  