import settingsPreparacion as settings
from utils.entradaSalida import cargaParametrosConfiguracionYAML, cargaImagen
from utils.utilidadesDirectorios import buscaFicheroMismoNombreBase
from utils.procesadoXML import getListaBndbox
from utils.graficosImagenes import dibujaRectangulos, creaListaRectangulos
from utils.dialogoFicheros import seleccionaFichero
import numpy as np
import matplotlib.pyplot as plt


def dibujaImagen(imagen):
    imArray = np.array(imagen)
    # Muestra la imagen usando matplotlib
    plt.imshow(imArray)
    plt.axis('on')  # Oculta los ejes
    plt.show()

def main():
    configuracion = cargaParametrosConfiguracionYAML(settings.PATH_PARAMETROS)
    if configuracion == None:
        print(f"Error cargando el fichero de configuración {settings.PATH_PARAMETROS}")
        return
    
    imagenPath = seleccionaFichero()
    if imagenPath == None:
        print('No se ha seleccionado ninguna imagen')
        return
    
    xmlPath = buscaFicheroMismoNombreBase(imagenPath, 'xml')
    if xmlPath == None:
        print(f'No se ha encontrado ningún archivo XML con el mismo nombre que la imagen seleccionada {imagenPath}.')
        return
    
    rectangulosHumanos = getListaBndbox(xmlPath)
    imagen = cargaImagen(imagenPath)

    subimageSize = configuracion['subimages']['size']
    overlap = configuracion['subimages']['overlap']
    margins = configuracion['subimages']['margins']
    listaRectangulos = creaListaRectangulos(imagen.size, subimageSize, overlap, margins)
    imagenGrid = dibujaRectangulos(imagen, listaRectangulos, color='black', ancho=2)
    imagenConRectangulos = dibujaRectangulos(imagenGrid, rectangulosHumanos)
    dibujaImagen(imagenConRectangulos)

if __name__ == '__main__':
    main()