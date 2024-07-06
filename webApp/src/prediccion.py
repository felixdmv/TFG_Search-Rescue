import os
# Evita la aparición del mensaje "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.src.saving.saving_api import load_model


def cargaModelo(ficheroModelo):
    """
    Loads a pre-trained model from a file.

    Parameters:
    ficheroModelo (str): The path to the model file.

    Returns:
    keras.models.Model: The loaded model.
    """
    return load_model(ficheroModelo, compile=False)


def cargaImagen1(ficheroImagen):
    """
    Carga una imagen desde un archivo.

    Args:
        ficheroImagen (str): La ruta del archivo de imagen a cargar.

    Returns:
        PIL.Image.Image: La imagen cargada.

    """
    imagen = load_img(ficheroImagen, target_size=None) # target_size= None --> Se deja al tamaño original
    return imagen


def cargaImagen(ficheroImagen):
    """
    Carga una imagen desde un archivo.

    Args:
        ficheroImagen (str): La ruta del archivo de imagen a cargar.

    Returns:
        PIL.Image.Image: La imagen cargada.

    """
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
        
    return salida