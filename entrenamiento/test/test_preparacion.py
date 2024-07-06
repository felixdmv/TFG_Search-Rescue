import os
import shutil
import tempfile
from PIL import Image
import pandas as pd
import pytest
from preparacion import (
    copiarImagenes, copiarImagenesDf, crearDirstrucTrain, crearDirstrucVal, crearDirstruc
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
    

@pytest.fixture
def setup_images():
    image_paths = []
    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, 'img')
    os.makedirs(img_dir)
    
    # Generate 10 images of size 10x10 pixels
    for i in range(10):
        img = Image.new('RGB', (10, 10), color='white')
        img_path = os.path.join(img_dir, f'image_{i}.png')
        img.save(img_path)
        image_paths.append(img_path)
    
    yield temp_dir, image_paths
    
    # Clean up
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

