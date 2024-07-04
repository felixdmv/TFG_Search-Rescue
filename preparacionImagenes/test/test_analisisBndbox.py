import pytest
from unittest.mock import patch, mock_open

from src.analisisBndbox import generaInformeBndbox



# Pruebas para la función generaInformeBndbox
@patch('builtins.open', new_callable=mock_open)
def test_generaInformeBndbox(mock_file):
    ficheroInforme = 'path/to/informe.txt'
    salidaAnalisis = (4, 20, 20, 10, 10, 15.0, 15.0)
    
    generaInformeBndbox(ficheroInforme, salidaAnalisis)
    
    # Verificar que se abrió el archivo correcto en modo escritura
    mock_file.assert_called_with(ficheroInforme, 'w', encoding='utf-8')
    
    # Obtener el manejador del archivo
    file_handle = mock_file()
    
    # Verificar que se escribió el contenido correcto
    file_handle.write.assert_any_call("Informe de Medidas de Bounding Boxes\n")
    file_handle.write.assert_any_call("===============================\n\n")
    file_handle.write.assert_any_call("Número total de Bounding Boxes: 4\n\n")
    file_handle.write.assert_any_call("Medidas:\n")
    file_handle.write.assert_any_call("- Ancho máximo: 20\n")
    file_handle.write.assert_any_call("- Alto máximo: 20\n")
    file_handle.write.assert_any_call("- Ancho mínimo: 10\n")
    file_handle.write.assert_any_call("- Alto mínimo: 10\n")
    file_handle.write.assert_any_call("- Ancho medio: 15.0\n")
    file_handle.write.assert_any_call("- Alto medio: 15.0\n\n")
    file_handle.write.assert_any_call("\nFin del Informe")

if __name__ == '__main__':
    pytest.main()
