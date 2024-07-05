# training.py
import time
import os
from PIL import ImageFile
import pandas as pd
from keras import layers

from preparacion import configurarTrainValGenerators
from redNeuronal import inicializarRed

def calcularPesosPorClase(trainGenerator):
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
 
# %%

# %%
def entrenamientoSimple(modelo, callbacks, trainGenerator, valGenerator, paramsRed):
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
    mejor = historico.iloc[[rowId-1]]
    
    return mejor


# In[ ]:
def pasada_uno(paths, dateTime, analisis, paramsRed, modelo, callbacks, trainGenerator, valGenerator):
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


# In[ ]:
def promediarModelos(modelo, nuevomodelo):
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


# In[ ]:
def pasadaIesima(numPasada, historico, bestPerformance, paths, dateTime, analisis, paramsRed, modelo, dirsEntrenamiento, totalSinUsar):
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
