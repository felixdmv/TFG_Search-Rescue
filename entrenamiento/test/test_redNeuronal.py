import os
import pytest
import shutil
import tempfile
import numpy as np
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import yaml
from settings_test import PATH_PARAMETROS
from src.redNeuronal import PrintLearningRate, MultiEarlyStopping
from src.redNeuronal import inicializarCallbacks, inicializarAlexnet, inicializarRed


@pytest.fixture(scope="module")
def paths_and_config():
    # Cargar el contenido del archivo YAML
    with open(PATH_PARAMETROS, 'r') as file:
        parametros = yaml.safe_load(file)
    
    paths = parametros.get('paths', {})
    analisis = parametros.get('analisis', {})
    redNeuronal = parametros.get('redNeuronal', {})
    
    # Generar dateTime actual
    dateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Definir parámetros de redNeuronal
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
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    callbacks = inicializarCallbacks(paths, dateTime, analisis, paramsRed, 'val_auc', 'max')
    
    tmp_dir_path = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime)
    assert os.path.exists(tmp_dir_path)
    
    res_dir_path = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
    assert os.path.exists(res_dir_path)
    
    assert isinstance(callbacks, list)
    assert len(callbacks) == 4

def test_inicializarAlexnet(paths_and_config):
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    model, monitor, mode = inicializarAlexnet(paths, dateTime, analisis, paramsRed)
    
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 15
    
    assert monitor == 'val_auroc'
    assert mode == 'max'

def test_inicializarRed(paths_and_config):
    paths, dateTime, analisis, paramsRed = paths_and_config
    
    model, callbacks = inicializarRed(paths, dateTime, analisis, paramsRed)
    
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 15
    
    assert isinstance(callbacks, list)
    assert len(callbacks) == 4

