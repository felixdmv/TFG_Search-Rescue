# training.py
import time
import os
from PIL import ImageFile
import pandas as pd
from keras import layers
from preparacion import configurarTrainValGenerators
from redNeuronal import inicializarRed


def calcularPesosPorClase(trainGenerator):
    """
    Calculates the class weights based on the number of images in each class.

    Args:
        trainGenerator: The training data generator.

    Returns:
        A dictionary containing the class weights, where the keys are the class indices and the values are the weights.
    """
    pathTraining = trainGenerator.directory
    clases = list(trainGenerator.class_indices.keys())

    directoriosBase = []
    for clase in clases:
        directoriosBase.append(os.path.join(pathTraining, clase))

    # Crear una lista para almacenar el número de imágenes por cada clase
    numImagenesPorClase = []
    
    # Iterar sobre los subdirectorios
    for subdirectorio in directoriosBase:
        archivos = os.listdir(subdirectorio)
        numImagenes = len(archivos)
        numImagenesPorClase.append(numImagenes)

        # Print para ver la ruta donde se encuentran las imágenes de cada directorio
        print(f"Directorio: {subdirectorio}")
        print(f"Número de imágenes en el directorio: {numImagenes}")

    # Print para ver el número total de imágenes
    totalImagenes = sum(numImagenesPorClase)
    print(f"Número total de imágenes: {totalImagenes}")

    # Cálculo de los pesos por clase
    classWeights = [totalImagenes / numero for numero in numImagenesPorClase]
    # Convertir la lista de pesos por clase a un diccionario
    classWeights = {indice: valor for indice, valor in enumerate(classWeights)}

    # Print para ver los pesos asignados a cada clase
    print("Pesos asignados a cada clase:")
    for clase, peso in zip(clases, classWeights.values()):
        print(f"Clase: {clase}, Peso: {peso}")

    return classWeights


def entrenamientoSimple(modelo, callbacks, trainGenerator, valGenerator, paramsRed):
    """
    Function to perform simple training of a model.

    Args:
        modelo (object): The model to be trained.
        callbacks (list): List of callback functions to be used during training.
        trainGenerator (object): The generator for training data.
        valGenerator (object): The generator for validation data.
        paramsRed (dict): Dictionary containing parameters for the model.

    Returns:
        tuple: A tuple containing the trained model, training history, and training duration in minutes.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    classWeights = calcularPesosPorClase(trainGenerator)
    
    # Guarda el tiempo de inicio
    inicio = time.time()
    
    history = modelo.fit(
        trainGenerator,
        epochs = paramsRed['epochs'],
        callbacks = callbacks,
        validation_data = valGenerator,
        class_weight = classWeights
    )
    
    # Guarda el tiempo de finalización
    fin = time.time()
    # Calculo del tiempo de entrenamiento en segundos
    duracionSegundos = fin - inicio
    # Convierte la duración a minutos
    duracionMinutos = duracionSegundos / 60

    historico = pd.DataFrame(history.history)
    historico.insert(0, 'lr', callbacks[0].learning_rates)

    return modelo, historico, duracionMinutos


def obtenerMejorDelHistorico(historico, rowId):
    """
    Returns the best record from the historical data based on the given row ID.

    Parameters:
    historico (pandas.DataFrame): The historical data.
    rowId (int): The ID of the row to retrieve.

    Returns:
    pandas.DataFrame: The best record from the historical data.
    """
    mejor = historico.iloc[[rowId-1]]
    
    return mejor


def pasada_uno(paths, dateTime, analisis, paramsRed, modelo, callbacks, trainGenerator, valGenerator):
    """
    Perform the first pass of training.

    Args:
        paths (dict): A dictionary containing file paths.
        dateTime (str): The current date and time.
        analisis (dict): A dictionary containing analysis parameters.
        paramsRed (dict): A dictionary containing network parameters.
        modelo (object): The model object.
        callbacks (list): A list of callback objects.
        trainGenerator (object): The training data generator object.
        valGenerator (object): The validation data generator object.

    Returns:
        tuple: A tuple containing the trained model, training history, and best performance.
    """
    tmpDir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime)

    print(f"PASADA 1 - CVITER {analisis['cviter']}")
    
    modelo, historico, _ = entrenamientoSimple(modelo, callbacks, trainGenerator, valGenerator, paramsRed)
    
    historico.insert(0, 'pasada', 1)
    bestPerformance = obtenerMejorDelHistorico(historico, callbacks[1].best_epoch)
    print(f"Mejor {paramsRed['metrica']} en validacion hasta ahora:")
    print(bestPerformance)
    
    archivosTmp = os.listdir(tmpDir)
    for archivo in archivosTmp:
        os.remove(os.path.join(tmpDir, archivo))

    #guardar modelo en disco
    fichero = os.path.join(tmpDir, f"{analisis['cviter']}.pasada1.modelo.h5")
    modelo.save(fichero)

    return modelo, historico, bestPerformance


def promediarModelos(modelo, nuevomodelo):
    """
    Promedia los pesos de las capas Dense de dos modelos y establece los pesos promediados en uno de los modelos.

    Args:
        modelo (tf.keras.Model): El primer modelo.
        nuevomodelo (tf.keras.Model): El segundo modelo.

    Returns:
        tf.keras.Model: El modelo con los pesos promediados establecidos.
    """
    capasModelo1 = modelo.layers
    capasDenseModelo1 = [capa for capa in capasModelo1 if isinstance(capa, layers.Dense)]
    capasModelo2 = nuevomodelo.layers
    capasDenseModelo2 = [capa for capa in capasModelo2 if isinstance(capa, layers.Dense)]

    pesosPromediados = []
    # Iterar sobre todas las capas de los modelos
    for capaModelo1, capaModelo2 in zip(capasDenseModelo1, capasDenseModelo2):
        # Obtener los pesos de ambas capas
        pesosModelo1 = capaModelo1.get_weights()
        pesosModelo2 = capaModelo2.get_weights()
            
        # Calcular el promedio de los pesos
        pesosPromediadosCapa = [(w1 + w2) / 2 for w1, w2 in zip(pesosModelo1, pesosModelo2)]
            
        # Agregar los pesos promediados a la lista
        pesosPromediados.append(pesosPromediadosCapa)
            
    # Establecer los pesos promediados en uno de los modelos
    for capa, pesosPromediadosCapa in zip(capasDenseModelo1, pesosPromediados):
        capa.set_weights(pesosPromediadosCapa)
        
    return modelo


def pasadaIesima(numPasada, historico, bestPerformance, paths, dateTime, analisis, paramsRed, modelo, dirsEntrenamiento, totalSinUsar):
    """
    Perform the i-th pass of training.

    Args:
        numPasada (int): The number of the current pass.
        historico (pd.DataFrame): The historical performance data.
        bestPerformance (pd.DataFrame): The best performance achieved so far.
        paths (dict): The paths to various directories.
        dateTime (str): The current date and time.
        analisis (dict): The analysis configuration.
        paramsRed (dict): The neural network parameters.
        modelo (tf.keras.Model): The current model.
        dirsEntrenamiento (dict): The training directories.
        totalSinUsar (int): The number of unused data samples.

    Returns:
        tf.keras.Model: The updated model.
        pd.DataFrame: The updated historical performance data.
        pd.DataFrame: The best performance achieved so far.
    """
    tmpDir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime)

    print()
    print(f"PASADA {numPasada} - CVITER {analisis['cviter']}")
    print(f"Quedan sin usar: {totalSinUsar}")
    
    trainGenerator, valGenerator = configurarTrainValGenerators(analisis, paramsRed, dirsEntrenamiento['train'], dirsEntrenamiento['val'])
    nuevomodelo, callbacks = inicializarRed(paths, dateTime, analisis, paramsRed, modelo)
    
    nuevomodelo, nuevoHistorico, _ = entrenamientoSimple(nuevomodelo, callbacks, trainGenerator, valGenerator, paramsRed)
    
    nuevoHistorico.insert(0, 'pasada', numPasada)
    newPerformance = obtenerMejorDelHistorico(nuevoHistorico, callbacks[1].best_epoch)
    historico = pd.concat([historico, nuevoHistorico], ignore_index=True)
        
    nuevaMetrica = 1 - newPerformance[f"val_{paramsRed['metrica']}"].iloc[0]  #harcodeado para metricas a maximizar
    mejorMetrica = 1 - bestPerformance[f"val_{paramsRed['metrica']}"].iloc[0] #harcodeado para metricas a maximizar
    nuevoLoss = newPerformance["val_loss"].iloc[0]
    mejorLoss = bestPerformance["val_loss"].iloc[0]

    if nuevaMetrica == 0:
        nuevaMetrica = 0.001
    elif nuevoLoss == 0:
        nuevoLoss = 0.001
    
    nuevoResultado = nuevaMetrica*nuevoLoss

    if mejorMetrica == 0:
        mejorMetrica = 0.001
    elif mejorLoss == 0:
        mejorLoss = 0.001
    
    mejorResultado = mejorMetrica*mejorLoss
    
    if  nuevoResultado <= mejorResultado:
        print("Se ha mejorado las metricas que se estan optimizando")
        bestPerformance = newPerformance
        modelo = nuevomodelo
    #elif nuevaMetrica > mejorMetrica or nuevoLoss < mejorLoss:
    #    print("Se mejora solo una de los dos metricas que se esta optimizando, promediamos el mejor modelo hasta ahora con el actual entrenado")
    #    modelo = promediarModelos(modelo, nuevomodelo)
        
    print(f"Mejor {paramsRed['metrica']} en validacion hasta ahora:")
    print(bestPerformance)

    archivosTmp = os.listdir(tmpDir)
    for archivo in archivosTmp:
        if "pasada" not in archivo:
            os.remove(os.path.join(tmpDir, archivo))

    #guardar modelo en disco
    fichero = os.path.join(tmpDir, f"{analisis['cviter']}.pasada{numPasada}.modelo.h5")
    modelo.save(fichero)
    os.remove(os.path.join(tmpDir, f"{analisis['cviter']}.pasada{numPasada-1}.modelo.h5"))

    return modelo, historico, bestPerformance