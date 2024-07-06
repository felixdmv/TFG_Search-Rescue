from PIL import Image
import pytest
from utils.graficosImagenes import creaListaRectangulos, creaListaRectangulosConIndices, dibujaRectangulos, dibujaRejilla


def test_creaListaRectangulos():
    """
    Test function for creaListaRectangulos.

    This function tests the creaListaRectangulos function by checking if the generated list of rectangles has the expected properties.

    Parameters:
    - imageSize: Tuple representing the size of the image (width, height).
    - subimageSize: Tuple representing the size of the subimage (width, height).
    - overlap: Tuple representing the overlap between subimages (horizontal, vertical).
    - margins: Tuple representing the margins around the image (horizontal, vertical).

    Returns:
    None
    """
    imageSize = (800, 600)
    subimageSize = (200, 150)
    overlap = (50, 50)
    margins = (100, 75)
    
    listaRectangulos = creaListaRectangulos(imageSize, subimageSize, overlap, margins)
    assert len(listaRectangulos) > 0
    for rect in listaRectangulos:
        assert len(rect) == 4  # Cada rectángulo debe tener 4 valores (left, upper, right, lower)


def test_creaListaRectangulosConIndices():
    """
    Test function for creaListaRectangulosConIndices.

    This function tests the creaListaRectangulosConIndices function by checking the following:
    - The length of the returned list is greater than 0.
    - Each rectangle in the list has 4 values (left, upper, right, lower).
    - Each index in the list is a tuple of two values.

    Returns:
        None
    """
    imageSize = (800, 600)
    subimageSize = (200, 150)
    overlap = (50, 50)
    margins = (100, 75)
    
    listaRectangulosConIndices = creaListaRectangulosConIndices(imageSize, subimageSize, overlap, margins)
    assert len(listaRectangulosConIndices) > 0
    for rect, idx in listaRectangulosConIndices:
        assert len(rect) == 4  # Cada rectángulo debe tener 4 valores (left, upper, right, lower)
        assert isinstance(idx, tuple) and len(idx) == 2  # Cada índice debe ser una tupla de dos valores


def test_dibujaRectangulos():
    """
    Test case for the dibujaRectangulos function.

    This function tests whether the dibujaRectangulos function correctly draws rectangles on an image.

    It creates a new image with a white background, defines a list of rectangles, and calls the dibujaRectangulos function
    to draw the rectangles on the image. Then, it asserts that the resulting image has the same size as the original image
    and that its mode is 'RGB'.

    This test case helps ensure that the dibujaRectangulos function behaves as expected and produces the desired output.

    """
    imagen = Image.new('RGB', (800, 600), (255, 255, 255))
    listaRectangulos = [(50, 50, 150, 150), (200, 200, 300, 300)]
    imagenConRectangulos = dibujaRectangulos(imagen, listaRectangulos)
    assert imagenConRectangulos.size == imagen.size  # La imagen con rectángulos debe tener el mismo tamaño que la original
    assert imagenConRectangulos.mode == 'RGB'


def test_dibujaRejilla():
    """
    Test case for the dibujaRejilla function.

    This function tests whether the dibujaRejilla function correctly draws a grid on an image.

    It creates a new image with a white background, defines a list of rectangles, and calls the dibujaRejilla function
    to draw a grid on the image. Then, it asserts that the resulting image has the same size as the original image and
    that its mode is 'RGB'.

    This test case helps ensure that the dibujaRejilla function behaves as expected and produces the desired output.

    """
    imagen = Image.new('RGB', (800, 600), (255, 255, 255))
    listaRectangulos = [(50, 50, 150, 150), (200, 200, 300, 300)]
    imagenConRejilla = dibujaRejilla(imagen, listaRectangulos)
    assert imagenConRejilla.size == imagen.size  # La imagen con la rejilla debe tener el mismo tamaño que la original
    assert imagenConRejilla.mode == 'RGB'


if __name__ == "__main__":
    pytest.main()