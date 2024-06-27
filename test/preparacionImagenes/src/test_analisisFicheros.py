import pytest
from unittest.mock import patch, mock_open

import settings
from settings import ROOT_TFG, ROOT_PREPARACIONIMAGENES, PATH_PARAMETROS, PATH_INFORMEBND, PATH_INFORMEANALISISFICHEROS
from preparacionImagenes.src.analisisFicheros import generaInformeAnalisisFicheros, main

# Fixture de mock para cargaParametrosConfiguracionYAML
@pytest.fixture
def mock_cargaParametrosConfiguracionYAML():
    with patch('utils.entradaSalida.cargaParametrosConfiguracionYAML') as mock:
        mock.return_value = {
            'dataSet': {
                'labelsSubfolder': 'labels'
            }
        }
        yield mock

# Fixture de mock para seleccionaDirectorio
@pytest.fixture
def mock_seleccionaDirectorio():
    with patch('utils.dialogoFicheros.seleccionaDirectorio') as mock:
        mock.return_value = '/path/to/dataset'
        yield mock

# Fixture de mock para open
@pytest.fixture
def mock_open_func():
    with patch('builtins.open', mock_open()) as m:
        yield m

@patch('builtins.open', new_callable=mock_open)
def test_generaInformeAnalisisFicheros(mock_file):
    ficheroInforme = str(settings.PATH_INFORMEANALISISFICHEROS)
    imagenesSinXML = {'imagen1', 'imagen2'}
    xmlSinImagen = {'archivo1'}
    
    # Llamamos a la función bajo prueba
    generaInformeAnalisisFicheros(ficheroInforme, imagenesSinXML, xmlSinImagen)
    
    # Verificar que se abrió el archivo correcto en modo escritura
    mock_file.assert_called_once_with(ficheroInforme, 'w', encoding='utf-8')
    
    # Obtener el manejador del archivo
    file_handle = mock_file()
    
    # Verificar que se escribieron las líneas esperadas
    expected_calls = [
        "Informe de Análisis de Ficheros\n",
        "=================================\n\n",
        "Archivos de imagen sin XML asociado:\n",
        "Se encontraron 2 archivos de imagen sin XML asociado.\n\n",
        "imagen1\n",
        "imagen2\n",
        "\n\nArchivos XML sin imagen asociada:\n",
        "Se encontraron 1 archivos XML sin imagen asociada.\n\n",
        "archivo1\n",
        "\nFin del Informe"
    ]
    
    # Obtener las llamadas reales al método write
    actual_calls = [call[0][0] for call in file_handle.write.call_args_list]
    
    # Verificar que todas las llamadas esperadas están presentes en las reales
    for call in expected_calls:
        assert call in actual_calls
