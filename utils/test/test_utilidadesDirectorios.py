import os
import shutil
import pytest
from utils.utilidadesDirectorios import (
    obtieneNombreBase,
    buscaFichero,
    buscaFicheroMismoNombreBase,
    obtienePathFromBasename,
    creaPathDirectorioMismoNivel,
    creaPathDirectorioNivelInferior,
    existePath,
    borraDirectorioYContenido,
    creaDirectorio,
    obtieneNombresBase,
    obtienePathFicheros
)


@pytest.fixture
def setup_test_files():
    """
    Creates a directory structure with test files for unit testing.

    Returns:
        str: The path to the directory containing the test files.
    """
    # Directory for test files
    test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    subdir1 = os.path.join(test_files_dir, "subdir1")
    subdir2 = os.path.join(test_files_dir, "subdir2")

    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)

    with open(os.path.join(subdir1, "file1.txt"), 'w') as f:
        f.write("Test file 1")

    with open(os.path.join(subdir1, "file1.xml"), 'w') as f:
        f.write("Test file 1 XML")

    with open(os.path.join(subdir2, "file2.txt"), 'w') as f:
        f.write("Test file 2")

    yield test_files_dir

    # Teardown
    shutil.rmtree(test_files_dir)


def test_obtieneNombreBase():
    """
    Test function for obtieneNombreBase.

    This function tests the behavior of the obtieneNombreBase function by asserting the expected output for different input cases.

    Input:
    - None

    Output:
    - None

    Returns:
    - None
    """
    assert obtieneNombreBase("/path/to/file.txt") == "file"
    assert obtieneNombreBase("file.txt") == "file"
    assert obtieneNombreBase("/path/to/file") == "file"


def test_buscaFichero(setup_test_files):
    """
    Test function for the buscaFichero function.

    Args:
        setup_test_files: The setup_test_files fixture.

    Returns:
        None

    Raises:
        AssertionError: If the test fails.
    """
    test_dir = setup_test_files
    assert buscaFichero(test_dir, "file1", "txt") == os.path.join(test_dir, "subdir1", "file1.txt")
    assert buscaFichero(test_dir, "file2", "txt") == os.path.join(test_dir, "subdir2", "file2.txt")
    assert buscaFichero(test_dir, "file1", "xml") == os.path.join(test_dir, "subdir1", "file1.xml")
    assert buscaFichero(test_dir, "file3", "txt") is None


def test_buscaFicheroMismoNombreBase(setup_test_files):
    """
    Test function for buscaFicheroMismoNombreBase.

    Args:
        setup_test_files: The setup_test_files fixture.

    Returns:
        None

    Raises:
        AssertionError: If the test fails.

    """
    test_dir = setup_test_files
    file_path = os.path.join(test_dir, "subdir1", "file1.txt")
    assert buscaFicheroMismoNombreBase(file_path, "xml") == os.path.join(test_dir, "subdir1", "file1.xml")
    assert buscaFicheroMismoNombreBase(file_path, "jpg") is None


def test_obtienePathFromBasename():
    """
    Test case for the obtienePathFromBasename function.

    This test case verifies that the obtienePathFromBasename function correctly
    returns the path obtained by concatenating the given directory, basename, and
    extension.

    It compares the normalized path returned by the function with the expected
    normalized path using the `os.path.normpath` function.

    Returns:
        None
    """
    assert os.path.normpath(obtienePathFromBasename("/path/to", "file", "txt")) == os.path.normpath("/path/to/file.txt")
    assert os.path.normpath(obtienePathFromBasename("path", "file", "jpg")) == os.path.normpath("path/file.jpg")


def test_creaPathDirectorioMismoNivel():
    """
    Test case for the creaPathDirectorioMismoNivel function.

    This function tests the behavior of the creaPathDirectorioMismoNivel function
    by asserting that the returned path is correct when creating a new directory
    at the same level as the given file path.

    The function compares the normalized paths of the expected result and the actual
    result using the os.path.normpath function.

    Example:
        assert os.path.normpath(creaPathDirectorioMismoNivel("/path/to/file.txt", "newdir")) == os.path.normpath("/path/to/newdir")
        assert os.path.normpath(creaPathDirectorioMismoNivel("C:\\path\\to\\file.txt", "newdir")) == os.path.normpath("C:\\path\\to\\newdir")
    """
    assert os.path.normpath(creaPathDirectorioMismoNivel("/path/to/file.txt", "newdir")) == os.path.normpath("/path/to/newdir")
    assert os.path.normpath(creaPathDirectorioMismoNivel("C:\\path\\to\\file.txt", "newdir")) == os.path.normpath("C:\\path\\to\\newdir")


def test_creaPathDirectorioNivelInferior():
    """
    Test case for the creaPathDirectorioNivelInferior function.

    This function tests the behavior of the creaPathDirectorioNivelInferior function by asserting that the normalized path returned by the function matches the expected normalized path.

    Returns:
        None
    """
    assert os.path.normpath(creaPathDirectorioNivelInferior("/path/to", "subdir")) == os.path.normpath("/path/to/subdir")
    assert os.path.normpath(creaPathDirectorioNivelInferior("C:\\path\\to", "subdir")) == os.path.normpath("C:\\path\\to\\subdir")


def test_existePath(setup_test_files):
    """
    Test the existePath function.

    Args:
        setup_test_files: A fixture that sets up the test files.

    Returns:
        None
    """
    test_dir = setup_test_files
    assert existePath(test_dir) is True
    assert existePath(os.path.join(test_dir, "nonexistent")) is False


def test_borraDirectorioYContenido(setup_test_files):
    """
    Test function to verify the behavior of the `borraDirectorioYContenido` function.

    Args:
        setup_test_files: A fixture that sets up the test files and returns the test directory.

    Returns:
        None
    """
    test_dir = setup_test_files
    temp_dir = os.path.join(test_dir, "temp")
    os.makedirs(temp_dir)
    assert existePath(temp_dir) is True
    borraDirectorioYContenido(temp_dir)
    assert existePath(temp_dir) is False


def test_creaDirectorio():
    """
    Test case for the creaDirectorio function.

    This function tests the functionality of the creaDirectorio function by creating a temporary directory,
    checking if the directory exists, and then deleting the directory and its contents.

    """
    temp_dir = "temp_test_dir"
    creaDirectorio(temp_dir)
    assert existePath(temp_dir) is True
    borraDirectorioYContenido(temp_dir)


def test_obtieneNombresBase(setup_test_files):
    """
    Test function for the obtieneNombresBase function.

    Args:
        setup_test_files: The setup_test_files fixture.

    Returns:
        None
    """
    test_dir = setup_test_files
    subdir1 = os.path.join(test_dir, "subdir1")
    assert set(obtieneNombresBase(subdir1)) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["txt"])) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["xml"])) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["jpg"])) == set()


def test_obtienePathFicheros(setup_test_files):
    """
    Test function for the obtienePathFicheros function.

    Args:
        setup_test_files: The setup_test_files fixture.

    Returns:
        None
    """
    test_dir = setup_test_files
    subdir1 = os.path.join(test_dir, "subdir1")
    expected_files = {os.path.join(subdir1, "file1.txt"), os.path.join(subdir1, "file1.xml")}
    assert set(obtienePathFicheros(subdir1)) == expected_files
    assert set(obtienePathFicheros(subdir1, ["txt"])) == {os.path.join(subdir1, "file1.txt")}
    assert set(obtienePathFicheros(subdir1, ["xml"])) == {os.path.join(subdir1, "file1.xml")}
    assert set(obtienePathFicheros(subdir1, ["jpg"])) == set()