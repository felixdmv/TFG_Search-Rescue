import os
import tempfile
from datetime import datetime
import pytest
import shutil
import pandas as pd
from unittest.mock import MagicMock
from keras import layers, models
from entrenamiento import calcularPesosPorClase, obtenerMejorDelHistorico, promediarModelos
from mainEntrenamiento import loadConfiguration 
from settingsEntrenamiento_test import PATH_PARAMETROS


configuracion = loadConfiguration(PATH_PARAMETROS)
paramsRed = configuracion['redNeuronal']
analisis = configuracion['analisis']
paths = configuracion['paths']


@pytest.fixture(scope='module')
def setup_data():
    """
    Sets up the data for training and validation.

    Returns:
        A tuple containing the following:
        - paths: A dictionary containing the paths for temporary and result directories.
        - dateTime: The current date and time in the format "%Y%m%d_%H%M%S".
        - analisis: The analysis object.
        - paramsRed: The parameters for the neural network.
        - directorios: A dictionary containing the paths for the train and validation directories.
    """
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
    """
    Test function for calcularPesosPorClase.

    Args:
        setup_data: A tuple containing the necessary setup data.

    Returns:
        None
    """
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
    """
    Test function to check the behavior of obtenerMejorDelHistorico function.

    This function creates a test DataFrame 'historico' with 'val_accuracy' and 'val_loss' columns.
    It then calls the obtenerMejorDelHistorico function with the 'historico' DataFrame and 3 as parameters.
    Finally, it asserts that the first row of the resulting DataFrame has 'val_accuracy' equal to 0.9
    and 'val_loss' equal to 0.3.
    """
    historico = pd.DataFrame({
        'val_accuracy': [0.8, 0.85, 0.9],
        'val_loss': [0.5, 0.4, 0.3]
    })

    mejor = obtenerMejorDelHistorico(historico, 3)

    assert mejor['val_accuracy'].iloc[0] == 0.9
    assert mejor['val_loss'].iloc[0] == 0.3


def test_promediarModelos():
    """
    Test function for promediarModelos.

    This function tests the promediarModelos function by creating two models,
    compiling them, and then averaging them using the promediarModelos function.
    The function checks if the output is an instance of models.Sequential.

    Returns:
        None
    """
    model1 = models.Sequential([layers.Dense(10, activation='relu', input_shape=(10,))])
    model1.add(layers.Dense(2, activation='softmax'))

    model2 = models.Sequential([layers.Dense(10, activation='relu', input_shape=(10,))])
    model2.add(layers.Dense(2, activation='softmax'))

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model1 = promediarModelos(model1, model2)

    assert isinstance(model1, models.Sequential)