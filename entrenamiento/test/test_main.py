import numpy as np
import pandas as pd
import pytest
import yaml
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
import tensorflow as tf
from src.main import loadConfiguration, setupGPU, saveResults, cleanup, main
from settings_test import PATH_PARAMETROS



def test_loadConfiguration():
    config = loadConfiguration(PATH_PARAMETROS)
    assert isinstance(config, dict)
    assert 'analisis' in config
    assert 'paths' in config
    assert 'redNeuronal' in config


def test_setupGPU():
    pass # No se puede probar porque no se puede acceder a la GPU en un entorno de pruebas


def test_saveResults():
    pass # No se puede probar ya que no se puede acceder al generador de imágenes y sus nuevas etiquetas
        
        
def test_cleanup(tmpdir):
    dateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    analisis = {'objetivo': 'test'}
    paths = {'datosEntreno': str(tmpdir), 'temporal': str(tmpdir)}

    train_path = tmpdir.mkdir(dateTime)
    temp_path = tmpdir.mkdir(analisis['objetivo']).mkdir(dateTime)  # Asegura que el directorio 'test' existe
    
    cleanup(paths, dateTime, analisis)
    
    assert not os.path.exists(train_path)
    assert not os.path.exists(temp_path)

    
def test_main():
    pass # No contiene código para probar
    
if __name__ == "__main__":
    pytest.main()