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
from settingsEntrenamiento import PATH_PARAMETROS


def loadConfiguration(configFile):
    """
    Loads the configuration from a YAML file.

    Args:
        configFile (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration.

    """
    with open(configFile, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setupGPU(analisis):
    """
    Sets up the GPU for training.

    Args:
        analisis (dict): A dictionary containing the analysis parameters.

    Returns:
        None
    """
    if analisis['idGpu'] >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[analisis['idGpu']], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[analisis['idGpu']], True)
        print(f"ELEGIDA LA GPU Nº {analisis['idGpu']}")


def saveResults(modelo, directorios, analisis, paramsRed, thresholds, resDir, tiempo):
    """
    Save the trained model and the test results to files.

    Args:
        modelo (keras.Model): The trained model to be saved.
        directorios (dict): A dictionary containing the directories for training and testing data.
        analisis (dict): A dictionary containing analysis parameters.
        paramsRed (dict): A dictionary containing network parameters.
        thresholds (dict): A dictionary containing threshold values.
        resDir (str): The directory where the results will be saved.
        tiempo (float): The time taken for training in minutes.
    """
    fichero = os.path.join(resDir, f"AlexNet.{analisis['cviter']}.modelo.h5")
    modelo.save(fichero)
    
    resultadosTest = calcularRendimientoTest(modelo, directorios['test'], analisis, paramsRed, thresholds)
    resultadosTest.insert(0, 'Iteracion', analisis['cviter'])
    resultadosTest.insert(0, 'Problema', analisis['objetivo'])
    resultadosTest.insert(len(resultadosTest.columns), 'Tiempo (mins)', tiempo)

    print(resultadosTest)
    resultadosTest.to_csv(os.path.join(resDir, f"AlexNet.{analisis['cviter']}.resultados.csv"), index=False)


def cleanup(paths, dateTime, analisis):
    """
    Removes the specified directories from the file system.

    Args:
        paths (dict): A dictionary containing the paths to the directories.
        dateTime (str): The name of the directory to be removed from the 'datosEntreno' path.
        analisis (dict): A dictionary containing the analysis information.

    Returns:
        bool: True if the directories were successfully removed, False otherwise.
    """
    try:
        shutil.rmtree(os.path.join(os.path.abspath(paths['datosEntreno']), dateTime))
        shutil.rmtree(os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime))
        return True
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False


def main():
    """
    Main function for training a neural network model.

    This function performs the following steps:
    1. Loads the configuration from a file.
    2. Sets up the GPU configuration if specified.
    3. Creates a data structure for analysis.
    4. Initializes the model and callbacks.
    5. Configures data generators for training and validation.
    6. Performs training based on the specified strategy.
    7. Saves the trained model and evaluates its performance.
    8. Saves the configuration to a file.
    9. Cleans up temporary directories.
    10. Checks if the results directory was created successfully.

    Note: The implementation for iterative training is not yet implemented.

    Returns:
        None
    """
    now = datetime.now()
    dateTime = now.strftime("%Y%m%d_%H%M%S")
    
    # Nombre del script
    nombreScript = 'main.py'
    
    # Load configuration
    configFile = PATH_PARAMETROS
    print(f"Using configuration file: {configFile}")  # Línea añadida para imprimir la ruta de PATH_PARAMETROS
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

    # Cleanup temporary directories and check if successful
    cleanup_success = cleanup(paths, dateTime, analisis)
    if cleanup_success:
        print("Temporary directories cleaned up successfully.")
    else:
        print("Failed to clean up temporary directories.")

    # Check if results directory exists
    if os.path.exists(resDir):
        print(f"Results directory {resDir} created successfully.")
    else:
        print(f"Failed to create results directory {resDir}.")


if __name__ == "__main__":
    main()