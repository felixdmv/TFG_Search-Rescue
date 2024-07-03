# main.py
import os
import shutil
import yaml
from datetime import datetime
import tensorflow as tf

from preparacion import dataStructureForAnalysisDroneSAR, configurarTrainValGenerators
from redNeuronal import inicializarRed
from entrenamiento import entrenamientoSimple
from metricas import calcularRendimientoTest

def loadConfiguration(configFile):
    with open(configFile, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setupGPU(analisis):
    if analisis['idGpu'] >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[analisis['idGpu']], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[analisis['idGpu']], True)
        print(f"ELEGIDA LA GPU Nº {analisis['idGpu']}")


def saveResults(modelo, directorios, analisis, paramsRed, thresholds, resDir, tiempo):
    fichero = os.path.join(resDir, f"AlexNet.{analisis['cviter']}.modelo.h5")
    modelo.save(fichero)
    
    resultadosTest = calcularRendimientoTest(modelo, directorios['test'], analisis, paramsRed, thresholds)
    resultadosTest.insert(0, 'Iteracion', analisis['cviter'])
    resultadosTest.insert(0, 'Problema', analisis['objetivo'])
    resultadosTest.insert(len(resultadosTest.columns), 'Tiempo (mins)', tiempo)

    print(resultadosTest)
    resultadosTest.to_csv(os.path.join(resDir, f"AlexNet.{analisis['cviter']}.resultados.csv"), index=False)


def cleanup(paths, dateTime, analisis):
    shutil.rmtree(os.path.join(os.path.abspath(paths['datosEntreno']), dateTime))
    shutil.rmtree(os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime))


def main():
    now = datetime.now()
    dateTime = now.strftime("%Y%m%d_%H%M%S")
    
    # Nombre del script
    nombreScript = 'main.py'
    
    # Load configuration
    configFile = './parametros.yaml'
    configuracion = loadConfiguration(configFile)
    
    configuracion['analisis']['script'] = nombreScript
    analisis    = configuracion['analisis']
    paths       = configuracion['paths']
    paramsRed   = configuracion['redNeuronal']

    # Set GPU configuration if specified
    setupGPU(analisis)

    # Create data structure for analysis
    directorios, imagenesPorUsar = dataStructureForAnalysisDroneSAR(os.path.abspath(paths['datosEntreno']), analisis, dateTime)

    # Initialize model and callbacks
    modelo, callbacks = inicializarRed(paths, dateTime, analisis, paramsRed)

    # Configure data generators
    trainGenerator, valGenerator = configurarTrainValGenerators(analisis, paramsRed, directorios['train'], directorios['val'])

    # Perform training based on strategy
    resDir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'], dateTime)

    if analisis['estrategiaEntreno'] == 'todo':
        modelo, _, tiempo = entrenamientoSimple(modelo, callbacks, trainGenerator, valGenerator, paramsRed)
    else:
        # Entrenamiento iterativo aún no implementado
        pass

    # Save model and evaluate performance
    saveResults(modelo, directorios, analisis, paramsRed, analisis["thresholds"], resDir, tiempo)

    # Save configuration
    fichero = os.path.join(resDir, f"AlexNet.{analisis['cviter']}.params_config.yaml")
    with open(fichero, 'w') as file:
        yaml.dump(configuracion, file)

    # Cleanup temporary directories
    cleanup(paths, dateTime, analisis)

if __name__ == "__main__":
    main()