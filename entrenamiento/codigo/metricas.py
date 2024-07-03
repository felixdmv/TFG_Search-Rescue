# metrics.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras

def configurarMetrica(metrica):
    if metrica == 'aupr':
        metricas = [keras.metrics.PrecisionAtRecall(num_thresholds=1000, recall=0.9, name="prec@rec"),
                    keras.metrics.RecallAtPrecision(num_thresholds=1000, precision=0.9, name="rec@prec"),
                    keras.metrics.AUC(num_thresholds=1000, curve="PR", name=metrica),
                    keras.metrics.AUC(num_thresholds=1000, name="auroc")]
        monitor = "val_aupr"
        mode = "max"
    elif metrica == 'auroc':
        metricas = [keras.metrics.PrecisionAtRecall(num_thresholds=1000, recall=0.9, name="prec@rec"),
                    keras.metrics.RecallAtPrecision(num_thresholds=1000, precision=0.9, name="rec@prec"),
                    keras.metrics.AUC(num_thresholds=1000, curve="PR", name="aupr"),
                    keras.metrics.AUC(num_thresholds=1000, name=metrica)]
        monitor = "val_auroc"
        mode = "max"
    elif metrica == 'rec@prec':
        metricas = [keras.metrics.PrecisionAtRecall(num_thresholds=1000, recall=0.9, name="prec@rec"),
                    keras.metrics.RecallAtPrecision(num_thresholds=1000, precision=0.9, name=metrica),
                    keras.metrics.AUC(num_thresholds=1000, curve="PR", name="aupr"),
                    keras.metrics.AUC(num_thresholds=1000, name="auroc")]
        monitor = "val_rec@prec"
        mode = "max"
    elif metrica == 'prec@rec':
        metricas = [keras.metrics.PrecisionAtRecall(num_thresholds=1000, recall=0.9, name=metrica),
                    keras.metrics.RecallAtPrecision(num_thresholds=1000, precision=0.9, name="rec@prec"),
                    keras.metrics.AUC(num_thresholds=1000, curve="PR", name="aupr"),
                    keras.metrics.AUC(num_thresholds=1000, name="auroc")]
        monitor = "val_prec@rec"
        mode = "max"

    return metricas, monitor, mode


def metricasBinclass(yReal, yPred, threshold=0.5):
    '''Funcion que calcula las metricas de rendimiento habituales para un clasificador binario.
    
    Parametros
    ----------
    yReal : np.array de 1 dimension
        Etiquetas reales, cada una representada por un número natural, de cada una de las muestras.
    yPred : np.array de 1 dimension
        Etiquetas predichas, representadas por una probabilidad entre 0 y 1, de cada una de las muestras.
        
    Retorno: una lista con las metricas [N, % 0's, % 1's, TN, FN, TP, FP, Accuracy, NPV, PPV, TNR, TPR, F1Score, AU-PR, AU-ROC]
    '''
    
    porcentajeUnos = np.mean(yReal) * 100
    porcentajeCeros = 100 - porcentajeUnos

    #umbral = obtener_mejor_threshold(yReal, yPred, thres_criteria)
    umbral = threshold
    
    # Binarizar el vector de probabilidades
    yPredBinario = np.where(yPred >= umbral, 1, 0)

    # Calcular la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(yReal, yPredBinario).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    npv = tn / (tn + fn + np.finfo(float).eps) #negative predictive value
    ppv = tp / (tp + fp + np.finfo(float).eps) #tambien conocido como precision
    tnr = tn / (tn + fp + np.finfo(float).eps) #tambien conocida como especificidad, la proporción de verdaderos negativos sobre el total de instancias negativas reales
    tpr = tp / (tp + fn + np.finfo(float).eps) #tambien conocida como sensibilidad o recall, la proporción de verdaderos positivos sobre el total de instancias positivas reales
    f1 = (2 * ppv * tpr) / (ppv + tpr  + np.finfo(float).eps)

    precision, recall, thresholds = precision_recall_curve(yReal, yPred)
    au_pr  = auc(recall, precision)
    au_roc = roc_auc_score(yReal, yPred)

    metricas = [sum([tn,fp,fn,tp]), porcentajeCeros, porcentajeUnos, umbral, tn, fn, tp, fp, accuracy, npv, ppv, tnr, tpr, f1, au_pr, au_roc]
    
    return metricas, yPredBinario


def calcularRendimientoTest(modelo, testDir, analisis, paramsRed, thresholds=[0.5]):
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )
    
    columnas = ['N', "% 0's", "% 1's", 'Umbral', 'TN', 'FN', 'TP', 'FP', 'Accuracy', 'NPV', 'PPV', 'TNR', 'TPR', 'F1Score', 'AU-PR', 'AU-ROC']
    dfResults = pd.DataFrame(columns=columnas)
    
    # Ajusta el tamaño de la imagen para AlexNet
    targetSize = (340, 340)  # Tamaño de entrada de AlexNet
    
    # Genera el generador de imágenes de prueba
    testGenerator = test_datagen.flow_from_directory(
        testDir,
        target_size=targetSize,
        color_mode=paramsRed['color'],
        batch_size=paramsRed['testBatchSize'],
        class_mode='binary',  # Siempre binaria para AlexNet
        classes = analisis['clases'],
        shuffle=False)

    # Obtiene las etiquetas verdaderas de prueba
    yTestTrue = testGenerator.labels
    # Obtiene las predicciones del modelo para las imágenes de prueba
    yTestProb = modelo.predict(testGenerator)
    
    # Calcula las métricas de rendimiento
    resultado = metricasBinclass(yTestTrue, yTestProb)

    # Iterar sobre los umbrales para calcular las métricas
    for thres in thresholds:
        resultado = metricasBinclass(yTestTrue, yTestProb, thres)
        dfResults.loc[len(dfResults)] = resultado[0]
    
    return dfResults



