import os
import pytest
import shutil
from datetime import datetime
from tensorflow import keras
import yaml
from settingsEntrenamiento_test import PATH_PARAMETROS
from redNeuronal import inicializarCallbacks, inicializarAlexnet, inicializarRed
from mainEntrenamiento import loadConfiguration


@pytest.fixture(scope="module")
def paths_and_config():
    """
    This function loads the content of a YAML file, extracts the necessary parameters,
    generates the current date and time, defines the neural network parameters, and sets
    the temporary and results directories based on the YAML configuration.

    Returns:
        A generator that yields the following values:
        - paths: A dictionary containing the paths extracted from the YAML file.
        - dateTime: A string representing the current date and time.
        - analisis: A dictionary containing the analysis parameters extracted from the YAML file.
        - paramsRed: A dictionary containing the neural network parameters extracted from the YAML file.
    """
    parametros = loadConfiguration(PATH_PARAMETROS)    
    paths = parametros.get('paths', {})
    analisis = parametros.get('analisis', {})
    redNeuronal = parametros.get('redNeuronal', {})
    
    # Genera dateTime actual
    dateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define parámetros de redNeuronal
    paramsRed = {
        'metrica': redNeuronal.get('metrica', 'accuracy'),
        'learningRate': redNeuronal.get('learningRate', 0.001),
        'decaySteps': redNeuronal.get('decaySteps', 1000),
        'decayRate': redNeuronal.get('decayRate', 0.5),
        'patience': redNeuronal.get('patience', 2),
        'startFromEpoch': redNeuronal.get('startFromEpoch', 0),
        'capasDensas': redNeuronal.get('capasDensas', 1),
        'neuronasPorCapa': redNeuronal.get('neuronasPorCapa', [4000]),
        'activacionPorCapa': redNeuronal.get('activacionPorCapa', ['relu']),
        'dropoutPorCapa': redNeuronal.get('dropoutPorCapa', [0]),
        # Otros parámetros de configuración de redNeuronal según sea necesario
    }
    
    # Establecer los directorios temporales y de resultados según el YAML
    tmp_dir = paths.get('temporal', '../tmp')
    res_dir = paths.get('resultados', '../resultados')
    
    yield paths, dateTime, analisis, paramsRed
    
    # Limpiar directorios temporales creados
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
        

def test_inicializarCallbacks(paths_and_config):
    """
    Test function to verify the behavior of the `inicializarCallbacks` function.

    Args:
        paths_and_config (tuple): A tuple containing the paths, dateTime, analisis, and paramsRed.

    Returns:
        None
    """
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    callbacks = inicializarCallbacks(paths, dateTime, analisis, paramsRed, 'val_auc', 'max')
    
    tmp_dir_path = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime)
    assert os.path.exists(tmp_dir_path)
    
    res_dir_path = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
    assert os.path.exists(res_dir_path)
    
    assert isinstance(callbacks, list)
    assert len(callbacks) == 4


def test_inicializarAlexnet(paths_and_config):
    """
    Test function to check the initialization of AlexNet model.

    Args:
        paths_and_config (tuple): A tuple containing paths, dateTime, analisis, and paramsRed.

    Returns:
        None
    """
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    model, monitor, mode = inicializarAlexnet(paths, dateTime, analisis, paramsRed)
    
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 16 # Hay que tener en cuenta las capas densas incluidas en el archivo YAML
    
    assert monitor == 'val_auroc'
    assert mode == 'max'

def test_inicializarRed(paths_and_config):
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    model, callbacks = inicializarRed(paths, dateTime, analisis, paramsRed)
    
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 16 # Hay que tener en cuetna las capas densas incluidas en el archivo YAML
    
    assert isinstance(callbacks, list)
    assert len(callbacks) == 4