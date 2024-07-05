import os
import tempfile
from datetime import datetime
import pytest
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch
from keras import layers, models
from PIL import ImageFile
from src.entrenamiento import calcularPesosPorClase, entrenamientoSimple, obtenerMejorDelHistorico, promediarModelos, pasada_uno, pasadaIesima
from settings_test import PATH_PARAMETROS
import yaml

# Cargar configuración desde YAML
with open(PATH_PARAMETROS, 'r') as file:
    configuracion = yaml.safe_load(file)

paramsRed = configuracion['redNeuronal']
analisis = configuracion['analisis']
paths = configuracion['paths']

@pytest.fixture(scope='module')
def setup_data():
    now = datetime.now()
    dateTime = now.strftime("%Y%m%d_%H%M%S")

    temp_dir = tempfile.mkdtemp()
    results_dir = tempfile.mkdtemp()
    
    paths['temporal'] = temp_dir
    paths['resultados'] = results_dir

    directorios = {
        'train': os.path.join(temp_dir, 'train'),
        'val': os.path.join(temp_dir, 'val')
    }
    os.makedirs(directorios['train'])
    os.makedirs(directorios['val'])
    
    yield paths, dateTime, analisis, paramsRed, directorios

    shutil.rmtree(temp_dir)
    shutil.rmtree(results_dir)

def test_calcularPesosPorClase(setup_data):
    paths, dateTime, analisis, paramsRed, directorios = setup_data

    # Mock trainGenerator
    trainGenerator = MagicMock()
    trainGenerator.directory = directorios['train']
    trainGenerator.class_indices = {'class1': 0, 'class2': 1}

    os.makedirs(os.path.join(directorios['train'], 'class1'))
    os.makedirs(os.path.join(directorios['train'], 'class2'))
    
    # Crear archivos de imagen de prueba
    for i in range(10):
        with open(os.path.join(directorios['train'], 'class1', f'image_{i}.jpg'), 'w'):
            pass

    for i in range(5):
        with open(os.path.join(directorios['train'], 'class2', f'image_{i}.jpg'), 'w'):
            pass

    class_weights = calcularPesosPorClase(trainGenerator)
    
    assert class_weights[0] == 1.5
    assert class_weights[1] == 3.0


def test_obtenerMejorDelHistorico():
    historico = pd.DataFrame({
        'val_accuracy': [0.8, 0.85, 0.9],
        'val_loss': [0.5, 0.4, 0.3]
    })

    mejor = obtenerMejorDelHistorico(historico, 3)

    assert mejor['val_accuracy'].iloc[0] == 0.9
    assert mejor['val_loss'].iloc[0] == 0.3

def test_promediarModelos():
    model1 = models.Sequential([layers.Dense(10, activation='relu', input_shape=(10,))])
    model1.add(layers.Dense(2, activation='softmax'))

    model2 = models.Sequential([layers.Dense(10, activation='relu', input_shape=(10,))])
    model2.add(layers.Dense(2, activation='softmax'))

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model1 = promediarModelos(model1, model2)

    assert isinstance(model1, models.Sequential)