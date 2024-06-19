# Importa settings.py para configurar los paths
import settings
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
    configuracion = cargaParametrosConfiguracionYAML('preparacionImagenes/config/parametros.yaml') # Ejecución desde el directorio raíz
    #configuracion = cargaParametrosConfiguracionYAML('../config/parametros.yaml') # Ejecución al mismo nivel que el script
    if configuracion == None:
        print('No se ha podido cargar el archivo de configuración "../config/parametros.yaml"')
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