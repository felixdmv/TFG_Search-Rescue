import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from mainEntrenamiento import loadConfiguration, cleanup
from settingsEntrenamiento_test import PATH_PARAMETROS


def test_loadConfiguration():
    """
    Test case for the loadConfiguration function.

    This function tests whether the loadConfiguration function returns a dictionary
    and checks if the required keys ('analisis', 'paths', 'redNeuronal') are present in the returned dictionary.
    """
    config = loadConfiguration(PATH_PARAMETROS)
    assert isinstance(config, dict)
    assert 'analisis' in config
    assert 'paths' in config
    assert 'redNeuronal' in config
     
        
def test_cleanup(tmpdir):
    """
    Test function for the cleanup function.

    Args:
        tmpdir: A temporary directory provided by the testing framework.

    Returns:
        None
    """
    dateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    analisis = {'objetivo': 'test'}
    paths = {'datosEntreno': str(tmpdir), 'temporal': str(tmpdir)}

    train_path = tmpdir.mkdir(dateTime)
    temp_path = tmpdir.mkdir(analisis['objetivo']).mkdir(dateTime)  # Asegura que el directorio 'test' existe
    
    cleanup(paths, dateTime, analisis)
    
    assert not os.path.exists(train_path)
    assert not os.path.exists(temp_path)

    
def test_setupGPU():
    pass # No se puede probar porque no se puede acceder a la GPU en un entorno de pruebas


def test_saveResults():
    pass # No se puede probar ya que no se puede acceder al generador de imágenes y sus nuevas etiquetas

   
def test_main():
    pass # No contiene código para probar

    
if __name__ == "__main__":
    pytest.main()