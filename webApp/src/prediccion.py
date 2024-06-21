import os
# Evita la aparición del mensaje "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.src.saving.saving_api import load_model  # load_model en la bibliografía se importa de otras formas, que dan "aparente" error en VSC


def cargaModelo(ficheroModelo):
    return load_model(ficheroModelo, compile=False)

def cargaImagen1(ficheroImagen):
    imagen = load_img(ficheroImagen, target_size=None) # target_size= None --> Se deja al tamaño original
    return imagen

def cargaImagen(ficheroImagen):
    imagen = load_img(ficheroImagen, target_size=None) # target_size= None --> Se deja al tamaño original
    return imagen

def predice(modelo, imagen, listaRectangulos, batch_size=1):
    """
    Predicts the output for a given model, image, and list of rectangles.

    Args:
        modelo (object): The trained model used for prediction.
        imagen (object): The input image.
        listaRectangulos (list): A list of rectangles specifying the regions of interest in the image.

    Returns:
        numpy.ndarray: An array containing the predictions for each subimage.

    """
    img_array = img_to_array(imagen)/255.0  # Normaliza la imagen
    listaSubimagenes = [img_array[upper:lower, left:right] for left, upper, right, lower in listaRectangulos]  # Submatrices usando slicing
    
    salida = []
    for subimagen in listaSubimagenes:
        subimagen = np.expand_dims(subimagen, axis=0)
        salida.append(modelo.predict_on_batch(subimagen)[0])
    # batch_size controla cuántas subimágenes se procesan a la vez. Experimentalmente he visto que todas
    # de golpe parece que es más rápido, pero si hay muchas imágenes salta aviso de que no hay memoria.
    # La idea parece ser que para grandes volúmenes de datos, ajustar el batch_size permite que quepa en memoria.
    # En la nube falla incluso con batch_size bajoo, en local admite 100 sin problema
    # Se deja a 1 para intentar que no falle en la nube,pero no es estable  
    # return modelo.predict(np.array(listaSubimagenes), batch_size=batch_size, verbose=1)
    return salida