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
    assert obtieneNombreBase("/path/to/file.txt") == "file"
    assert obtieneNombreBase("file.txt") == "file"
    assert obtieneNombreBase("/path/to/file") == "file"

def test_buscaFichero(setup_test_files):
    test_dir = setup_test_files
    assert buscaFichero(test_dir, "file1", "txt") == os.path.join(test_dir, "subdir1", "file1.txt")
    assert buscaFichero(test_dir, "file2", "txt") == os.path.join(test_dir, "subdir2", "file2.txt")
    assert buscaFichero(test_dir, "file1", "xml") == os.path.join(test_dir, "subdir1", "file1.xml")
    assert buscaFichero(test_dir, "file3", "txt") is None

def test_buscaFicheroMismoNombreBase(setup_test_files):
    test_dir = setup_test_files
    file_path = os.path.join(test_dir, "subdir1", "file1.txt")
    assert buscaFicheroMismoNombreBase(file_path, "xml") == os.path.join(test_dir, "subdir1", "file1.xml")
    assert buscaFicheroMismoNombreBase(file_path, "jpg") is None

def test_obtienePathFromBasename():
    assert obtienePathFromBasename("/path/to", "file", "txt") == "/path/to/file.txt"
    assert obtienePathFromBasename("path", "file", "jpg") == "path/file.jpg"

def test_creaPathDirectorioMismoNivel():
    assert creaPathDirectorioMismoNivel("/path/to/file.txt", "newdir") == "/path/to/newdir"

def test_creaPathDirectorioNivelInferior():
    assert creaPathDirectorioNivelInferior("/path/to", "subdir") == "/path/to/subdir"

def test_existePath(setup_test_files):
    test_dir = setup_test_files
    assert existePath(test_dir) is True
    assert existePath(os.path.join(test_dir, "nonexistent")) is False

def test_borraDirectorioYContenido(setup_test_files):
    test_dir = setup_test_files
    temp_dir = os.path.join(test_dir, "temp")
    os.makedirs(temp_dir)
    assert existePath(temp_dir) is True
    borraDirectorioYContenido(temp_dir)
    assert existePath(temp_dir) is False

def test_creaDirectorio():
    temp_dir = "temp_test_dir"
    creaDirectorio(temp_dir)
    assert existePath(temp_dir) is True
    borraDirectorioYContenido(temp_dir)

def test_obtieneNombresBase(setup_test_files):
    test_dir = setup_test_files
    subdir1 = os.path.join(test_dir, "subdir1")
    assert set(obtieneNombresBase(subdir1)) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["txt"])) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["xml"])) == {"file1"}
    assert set(obtieneNombresBase(subdir1, ["jpg"])) == set()

def test_obtienePathFicheros(setup_test_files):
    test_dir = setup_test_files
    subdir1 = os.path.join(test_dir, "subdir1")
    expected_files = {os.path.join(subdir1, "file1.txt"), os.path.join(subdir1, "file1.xml")}
    assert set(obtienePathFicheros(subdir1)) == expected_files
    assert set(obtienePathFicheros(subdir1, ["txt"])) == {os.path.join(subdir1, "file1.txt")}
    assert set(obtienePathFicheros(subdir1, ["xml"])) == {os.path.join(subdir1, "file1.xml")}
    assert set(obtienePathFicheros(subdir1, ["jpg"])) == set()
