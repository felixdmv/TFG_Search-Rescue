from PIL import Image
import pytest
from utils.graficosImagenes import creaListaRectangulos, creaListaRectangulosConIndices, dibujaRectangulos, dibujaRejilla


def test_creaListaRectangulos():
    imageSize = (800, 600)
    subimageSize = (200, 150)
    overlap = (50, 50)
    margins = (100, 75)
    
    listaRectangulos = creaListaRectangulos(imageSize, subimageSize, overlap, margins)
    assert len(listaRectangulos) > 0
    for rect in listaRectangulos:
        assert len(rect) == 4  # Cada rectángulo debe tener 4 valores (left, upper, right, lower)

def test_creaListaRectangulosConIndices():
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
    imagen = Image.new('RGB', (800, 600), (255, 255, 255))
    listaRectangulos = [(50, 50, 150, 150), (200, 200, 300, 300)]
    imagenConRectangulos = dibujaRectangulos(imagen, listaRectangulos)
    assert imagenConRectangulos.size == imagen.size  # La imagen con rectángulos debe tener el mismo tamaño que la original
    assert imagenConRectangulos.mode == 'RGB'

def test_dibujaRejilla():
    imagen = Image.new('RGB', (800, 600), (255, 255, 255))
    listaRectangulos = [(50, 50, 150, 150), (200, 200, 300, 300)]
    imagenConRejilla = dibujaRejilla(imagen, listaRectangulos)
    assert imagenConRejilla.size == imagen.size  # La imagen con la rejilla debe tener el mismo tamaño que la original
    assert imagenConRejilla.mode == 'RGB'

if __name__ == "__main__":
    pytest.main()