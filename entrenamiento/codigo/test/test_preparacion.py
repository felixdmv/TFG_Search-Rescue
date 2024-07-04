import os
import shutil
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from preparacion import (
    copiarImagenes, copiarImagenesDf, copiarImagenesResto,
    crearDirstrucTrain, crearDirstrucVal, crearDirstruc,
    dataStructureForAnalysisDroneSAR, configurarTrainValGenerators
)

@pytest.fixture
def setup_directories():
    test_dir = './test_files'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    os.makedirs(os.path.join(test_dir, 'images', 'dataset1', 'dataset1'))
    os.makedirs(os.path.join(test_dir, 'images', 'dataset2', 'dataset2'))
    
    # Crear imágenes de prueba
    for i in range(5):
        open(os.path.join(test_dir, 'images', 'dataset1', 'dataset1', f'image_{i}.jpg'), 'a').close()
        open(os.path.join(test_dir, 'images', 'dataset2', 'dataset2', f'image_{i}.jpg'), 'a').close()
    
    yield test_dir
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_copiarImagenes(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg']
    })
    pathDestino = os.path.join(test_dir, 'destino')
    os.makedirs(pathDestino)
    
    copiarImagenes(pathImagenes, dfImagenes, pathDestino)
    
    assert os.path.exists(os.path.join(pathDestino, 'image_0.jpg'))
    assert os.path.exists(os.path.join(pathDestino, 'image_1.jpg'))

def test_copiarImagenesDf(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    analisis = {
        'objetivo': 'label',
        'clases': ['class1', 'class2']
    }
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg'],
        'label': [0, 1]
    })
    dir_destino = os.path.join(test_dir, 'destino')
    
    dframe, añadidas, total = copiarImagenesDf(pathImagenes, analisis, dir_destino, dfImagenes)
    
    assert os.path.exists(os.path.join(dir_destino, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(dir_destino, 'class2', 'image_1.jpg'))
    assert añadidas == [1, 1]
    assert total == 2


def test_copiarImagenesResto(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    analisis = {
        'objetivo': 'label',
        'clases': ['class1', 'class2']
    }
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2', 'dataset1', 'dataset2', 'dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg', 'image_5.jpg'],
        'label': [0, 1, 0, 1, 0, 1],
        'sin usar': [1, 1, 1, 1, 1, 1]
    })
    dir_general = os.path.join(test_dir, 'general')
    
    # Simulate local images count and class count
    numLocalImages = 2
    conteoLocalPorClase = [2, 2]
    
    # Creating initial directories and images for the test
    for idx, row in dfImagenes.iterrows():
        dataset_path = os.path.join(pathImagenes, row['Dataset'], row['Dataset'])
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        image_path = os.path.join(dataset_path, row['Nombre del archivo'])
        with open(image_path, 'w') as f:
            f.write('test image data')
    
    updated_dframe = copiarImagenesResto(pathImagenes, analisis, dir_general, dfImagenes, numLocalImages, conteoLocalPorClase)
    
    # Check if images were copied correctly to their respective directories
    assert os.path.exists(os.path.join(dir_general, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class2', 'image_1.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class1', 'image_2.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class2', 'image_3.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class1', 'image_4.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class2', 'image_5.jpg'))
    
    # Check if 'sin usar' column was updated correctly
    assert all(updated_dframe.loc[updated_dframe['Nombre del archivo'].isin(['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg', 'image_5.jpg']), 'sin usar'] == 0)


def test_crearDirstrucTrain(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    analisis = {
        'estrategiaEntreno': 'todo',
        'objetivo': 'label',
        'clases': ['class1', 'class2']
    }
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg'],
        'label': [0, 1]
    })
    trainDir = os.path.join(test_dir, 'train')
    
    crearDirstrucTrain(pathImagenes, analisis, trainDir, dfImagenes)
    
    assert os.path.exists(os.path.join(trainDir, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(trainDir, 'class2', 'image_1.jpg'))


def test_crearDirstrucVal(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    analisis = {
        'estrategiaEntreno': 'todo',
        'objetivo': 'label',
        'clases': ['class1', 'class2']
    }
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg'],
        'label': [0, 1]
    })
    valDir = os.path.join(test_dir, 'val')
    
    crearDirstrucVal(pathImagenes, analisis, valDir, dfImagenes)
    
    assert os.path.exists(os.path.join(valDir, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(valDir, 'class2', 'image_1.jpg'))

def test_crearDirstruc(setup_directories):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    analisis = {
        'objetivo': 'label',
        'clases': ['class1', 'class2']
    }
    dfImagenes = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg'],
        'label': [0, 1]
    })
    dir_general = os.path.join(test_dir, 'general')
    
    crearDirstruc(pathImagenes, analisis, dir_general, dfImagenes)
    
    assert os.path.exists(os.path.join(dir_general, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(dir_general, 'class2', 'image_1.jpg'))
    

def test_dataStructureForAnalysisDroneSAR(setup_directories, tmpdir):
    test_dir = setup_directories
    pathImagenes = os.path.join(test_dir, 'images')
    csv_file = os.path.join(test_dir, 'data.csv')
    
    df_csv = pd.DataFrame({
        'Dataset': ['dataset1', 'dataset2', 'dataset1', 'dataset2', 'dataset1'],
        'Nombre del archivo': ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg'],
        'label': [0, 1, 0, 1, 0],
        'Caja label': [1, 2, 1, 3, 1]
    })
    df_csv.to_csv(csv_file, index=False)
    
    analisis = {
        'ficheroCsv': csv_file,
        'objetivo': 'label',
        'clases': ['class1', 'class2'],
        'cviter': 1
    }
    dateTime = '2024-07-04'
    
    # Crear un archivo de configuración simulado
    config_content = """
    paths:
      imagenes: './test_files/images'
    """
    config_path = os.path.join(test_dir, 'parametros.yaml')
    with open(config_path, 'w') as config_file:
        config_file.write(config_content)
    
    with patch('preparation.yaml.safe_load', return_value={'paths': {'imagenes': pathImagenes}}):
        directorios, dframes = dataStructureForAnalysisDroneSAR(test_dir, analisis, dateTime)
    
    trainDir = directorios['train']
    valDir = directorios['val']
    testDir = directorios['test']
    
    assert os.path.exists(os.path.join(trainDir, 'class1', 'image_0.jpg'))
    assert os.path.exists(os.path.join(trainDir, 'class1', 'image_2.jpg'))
    assert os.path.exists(os.path.join(trainDir, 'class1', 'image_4.jpg'))
    assert os.path.exists(os.path.join(valDir, 'class2', 'image_1.jpg'))
    assert os.path.exists(os.path.join(testDir, 'class2', 'image_3.jpg'))
    
    assert len(dframes['train']) == 3
    assert len(dframes['val']) == 1
    assert len(dframes['test']) == 1
    
        
@patch('preparation.ImageDataGenerator')
def test_configurarTrainValGenerators(mock_ImageDataGenerator, setup_directories):
    test_dir = setup_directories
    analisis = {
        'dataAugmentation': 0,
        'clases': ['class1', 'class2']
    }
    paramsRed = {
        'color': 'rgb',
        'trainBatchSize': 32,
        'valBatchSize': 16
    }
    trainDir = os.path.join(test_dir, 'train')
    valDir = os.path.join(test_dir, 'val')
    
    mock_train_gen = MagicMock()
    mock_val_gen = MagicMock()
    mock_ImageDataGenerator.return_value = mock_train_gen
    mock_train_gen.flow_from_directory.return_value = 'train_gen'
    mock_val_gen.flow_from_directory.return_value = 'val_gen'
    
    trainGenerator, valGenerator = configurarTrainValGenerators(analisis, paramsRed, trainDir, valDir)
    
    mock_ImageDataGenerator.assert_called_with(rescale=1.0 / 255)
    mock_train_gen.flow_from_directory.assert_called_with(
        trainDir, target_size=(340, 340), color_mode='rgb',
        batch_size=32, class_mode='binary', classes=['class1', 'class2'], shuffle=True
    )
    mock_val_gen.flow_from_directory.assert_called_with(
        valDir, target_size=(340, 340), color_mode='rgb',
        batch_size=16, class_mode='binary', classes=['class1', 'class2'], shuffle=False
    )
    assert trainGenerator == 'train_gen'
    assert valGenerator == 'val_gen'

    # Test with dataAugmentation
    analisis['dataAugmentation'] = 1
    trainGenerator, valGenerator = configurarTrainValGenerators(analisis, paramsRed, trainDir, valDir)
    
    mock_ImageDataGenerator.assert_called_with(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
