import pytest
from unittest.mock import patch
from utils.dialogoFicheros import seleccionaDirectorio, seleccionaFichero


@patch('tkinter.filedialog.askdirectory')
@patch('tkinter.Tk')
def test_seleccionaDirectorio(mock_tk, mock_askdirectory):
    """
    Test case for the seleccionaDirectorio function.

    This test case verifies the behavior of the seleccionaDirectorio function
    when a directory is selected, when the selection is canceled, and also
    checks that the askdirectory dialog is called once and the Tkinter root
    is destroyed.

    Args:
        mock_tk (Mock): A mock object for the Tkinter root.
        mock_askdirectory (Mock): A mock object for the askdirectory function.

    Returns:
        None

    Raises:
        AssertionError: If the test case fails.

    """
    # Simular la selección de un directorio
    mock_askdirectory.return_value = '/path/to/directory'
    assert seleccionaDirectorio() == '/path/to/directory'
    
    # Simular la cancelación del diálogo de selección de directorio
    mock_askdirectory.return_value = ''
    assert seleccionaDirectorio() is None
    
    # Verificar que el diálogo de selección de directorio fue llamado una vez
    mock_askdirectory.assert_called()
    # Verificar que el Tkinter root fue destruido
    assert mock_tk.return_value.destroy.call_count == 2

@patch('tkinter.filedialog.askopenfilename')
@patch('tkinter.Tk')
def test_seleccionaFichero(mock_tk, mock_askopenfilename):
    """
    Test case for the seleccionaFichero function.

    This test case verifies the behavior of the seleccionaFichero function
    when a file is selected, when the file selection dialog is canceled,
    and when the Tkinter root is destroyed.

    Args:
        mock_tk (Mock): A mock object for the Tkinter root.
        mock_askopenfilename (Mock): A mock object for the askopenfilename function.

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Simular la selección de un archivo
    mock_askopenfilename.return_value = '/path/to/file.txt'
    assert seleccionaFichero() == '/path/to/file.txt'
    
    # Simular la cancelación del diálogo de selección de archivo
    mock_askopenfilename.return_value = ''
    assert seleccionaFichero() is None
    
    # Verificar que el diálogo de selección de archivo fue llamado una vez
    mock_askopenfilename.assert_called()
    # Verificar que el Tkinter root fue destruido
    assert mock_tk.return_value.destroy.call_count == 2


if __name__ == '__main__':
    pytest.main()