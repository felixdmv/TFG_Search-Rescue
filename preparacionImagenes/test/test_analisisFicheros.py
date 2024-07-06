import pytest
from unittest.mock import patch, mock_open
from analisisFicheros import generaInformeAnalisisFicheros


@patch('builtins.open', new_callable=mock_open)
def test_generaInformeAnalisisFicheros(mock_file):
    """
    Test case for the generaInformeAnalisisFicheros function.

    Args:
        mock_file: A mock object representing the file.

    Returns:
        None
    """
    ficheroInforme = 'prueba.txt' 
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