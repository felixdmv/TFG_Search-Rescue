import os
import pytest
import json
from unittest.mock import patch, mock_open
from PIL import Image
from utils.entradaSalida import cargaParametrosConfiguracionYAML, cargaArchivoDrive, cargaParametrosProcesamiento, rectangulosEtiquetados, cargaImagen


# Ruta de los archivos de prueba
test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
yaml_config_file = os.path.join(test_files_dir, 'config.yaml')
json_params_file = os.path.join(test_files_dir, 'parameters.json')
test_image_file = os.path.join(test_files_dir, 'test_image.jpg')
test_xml_file = os.path.join(test_files_dir, 'test_annotation.xml')

# Contenido del archivo YAML de prueba
yaml_content = """
param1: value1
param2: value2
param3:
  subparam1: subvalue1
  subparam2: subvalue2
"""

# Contenido del archivo JSON de prueba
json_content = {
    "param1": "value1",
    "param2": "value2",
    "param3": {
        "subparam1": "subvalue1",
        "subparam2": "subvalue2"
    }
}

# Contenido del archivo XML de prueba
xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotation>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>3471</xmin>
         <xmax>3540</xmax>
         <ymin>1195</ymin>
         <ymax>1275</ymax>
      </bndbox>
   </object>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>2645</xmin>
         <xmax>2709</xmax>
         <ymin>2820</ymin>
         <ymax>2887</ymax>
      </bndbox>
   </object>
   <folder>SAR</folder>
   <filename>BLA_0001</filename>
   <source>
      <database>HERIDAL database</database>
   </source>
   <size>
      <width>4000</width>
      <height>3000</height>
      <depth>3</depth>
   </size>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>355</xmin>
         <xmax>435</xmax>
         <ymin>1751</ymin>
         <ymax>1791</ymax>
      </bndbox>
   </object>
</annotation>
"""


@pytest.fixture(scope='module')
def setup_test_files():
    """
    Set up test files for unit testing.

    This function creates a directory for test files and generates various types of test files,
    including a YAML file, a JSON file, an XML file, and an image file.

    Returns:
        None
    """
    # Crear directorio de archivos de prueba
    os.makedirs(test_files_dir, exist_ok=True)
    
    # Crear archivo YAML de prueba
    with open(yaml_config_file, 'w') as f:
        f.write(yaml_content)
    
    # Crear archivo JSON de prueba
    with open(json_params_file, 'w') as f:
        json.dump(json_content, f)
    
    # Crear archivo XML de prueba
    with open(test_xml_file, 'w') as f:
        f.write(xml_content)

    # Crear una imagen de prueba
    image = Image.new('RGB', (100, 100), color='red')
    image.save(test_image_file)

    yield


def test_cargaParametrosConfiguracionYAML(setup_test_files):
    """
    Test case for the cargaParametrosConfiguracionYAML function.
    """
    config = cargaParametrosConfiguracionYAML(yaml_config_file)
    assert config['param1'] == 'value1'
    assert config['param2'] == 'value2'
    assert config['param3']['subparam1'] == 'subvalue1'
    assert config['param3']['subparam2'] == 'subvalue2'


def test_cargaArchivoDrive():
    """
    Test case for the cargaArchivoDrive function.

    This function tests the behavior of the cargaArchivoDrive function by
    providing a real URL and a fake URL. It verifies that the function
    successfully downloads the file from the real URL and fails to download
    the file from the fake URL.

    Raises:
        Exception: If the file download fails with the real URL.

    Returns:
        None
    """
    real_url = "https://drive.google.com/uc?id=1suPGFkdPGrPQdpsyyzrUH1EmNilq5ecP"
    fake_url = "https://drive.google.com/uc?id=fakeid1234"
    output_path = os.path.join(test_files_dir, 'downloaded_file.h5')
    
    # Probar con la URL real
    try:
        cargaArchivoDrive(real_url, output_path)
        assert os.path.exists(output_path)
        os.remove(output_path)
    except Exception as e:
        pytest.fail(f"Descarga fallida con URL real: {e}")

    # Probar con la URL falsa
    with pytest.raises(Exception):
        cargaArchivoDrive(fake_url, output_path)


def test_cargaParametrosProcesamiento(setup_test_files):
    """
    Test case for cargaParametrosProcesamiento function.

    This test case verifies that the cargaParametrosProcesamiento function correctly loads the parameters from a JSON file and asserts their values.

    Args:
        setup_test_files: A fixture that sets up the necessary test files.

    Returns:
        None
    """
    params = cargaParametrosProcesamiento(json_params_file)
    assert params['param1'] == 'value1'
    assert params['param2'] == 'value2'
    assert params['param3']['subparam1'] == 'subvalue1'
    assert params['param3']['subparam2'] == 'subvalue2'


def test_rectangulosEtiquetados(setup_test_files):
    """
    Test function for the rectangulosEtiquetados function.
    
    This function tests the rectangulosEtiquetados function by asserting that the returned
    rectangles match the expected rectangles.
    
    Args:
        setup_test_files: A setup fixture for the test.
    
    Returns:
        None
    """
    rects = rectangulosEtiquetados(test_xml_file)
    assert rects == [
        (3471, 1195, 3540, 1275),
        (2645, 2820, 2709, 2887),
        (355, 1751, 435, 1791)
    ]


def test_cargaImagen(setup_test_files):
    """
    Test the cargaImagen function.

    This function tests the cargaImagen function by asserting that the loaded image has the correct size and mode.

    Parameters:
    - setup_test_files: A fixture that sets up the necessary test files.

    Returns:
    - None
    """
    img = cargaImagen(test_image_file)
    assert img.size == (100, 100)
    assert img.mode == 'RGB'