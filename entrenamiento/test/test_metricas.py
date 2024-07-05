import pytest
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from unittest.mock import MagicMock, patch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.metricas import configurarMetrica, metricasBinclass, calcularRendimientoTest

@pytest.fixture(scope='module')
def setup_data():
    # Crear datos ficticios para las pruebas
    yReal = np.array([0, 0, 1, 1])
    yPred = np.array([0.1, 0.4, 0.35, 0.8])
    return yReal, yPred

def test_configurarMetrica():
    metricas, monitor, mode = configurarMetrica('aupr')
    assert monitor == "val_aupr"
    assert mode == "max"
    assert len(metricas) == 4
    
    metricas, monitor, mode = configurarMetrica('auroc')
    assert monitor == "val_auroc"
    assert mode == "max"
    assert len(metricas) == 4
    
    metricas, monitor, mode = configurarMetrica('rec@prec')
    assert monitor == "val_rec@prec"
    assert mode == "max"
    assert len(metricas) == 4
    
    metricas, monitor, mode = configurarMetrica('prec@rec')
    assert monitor == "val_prec@rec"
    assert mode == "max"
    assert len(metricas) == 4

def test_metricasBinclass(setup_data):
    yReal, yPred = setup_data
    metricas, yPredBinario = metricasBinclass(yReal, yPred, threshold=0.5)
    
    assert len(metricas) == 16
    assert isinstance(metricas, list)
    assert isinstance(yPredBinario, np.ndarray)

def test_metricasBinclass_values(setup_data):
    yReal, yPred = setup_data
    metricas, yPredBinario = metricasBinclass(yReal, yPred, threshold=0.5)
    
    # Verifica los valores de las métricas
    assert metricas[4] == 1  # TN
    assert metricas[5] == 1  # FN
    assert metricas[6] == 1  # TP
    assert metricas[7] == 1  # FP
    assert np.isclose(metricas[8], 0.5)  # Accuracy
    assert np.isclose(metricas[9], 0.5)  # NPV
    assert np.isclose(metricas[10], 0.5)  # PPV
    assert np.isclose(metricas[11], 0.5)  # TNR
    assert np.isclose(metricas[12], 0.5)  # TPR
    assert np.isclose(metricas[13], 0.5)  # F1Score
    assert np.isclose(metricas[14], 0.5833, atol=1e-4)  # AU-PR
    assert np.isclose(metricas[15], 0.75)  # AU-ROC

@pytest.fixture(scope='module')
def setup_model():
    # Crear un modelo simple para pruebas
    model = models.Sequential([layers.Dense(1, input_shape=(10,))])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def test_calcularRendimientoTest(setup_model, tmpdir):
    model = setup_model
    test_dir = tmpdir.mkdir('test_images')

    class1 = test_dir.mkdir('class1')
    class2 = test_dir.mkdir('class2')
    
    for i in range(5):
        class1.join(f'image_{i}.jpg').write('')
    
    for i in range(5):
        class2.join(f'image_{i}.jpg').write('')

    analisis = {'clases': ['class1', 'class2']}
    paramsRed = {'color': 'rgb', 'testBatchSize': 2}
    thresholds = [0.5]

    # Mock ImageDataGenerator y su flujo de datos
    with patch.object(ImageDataGenerator, 'flow_from_directory') as mock_flow:
        mock_flow.return_value = MagicMock()
        mock_flow.return_value.labels = np.array([0] * 5 + [1] * 5)
        mock_flow.return_value.__len__.return_value = 5
        mock_flow.return_value.__getitem__.return_value = (np.zeros((2, 340, 340, 3)), np.zeros(2))

        with patch.object(model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([0.1] * 5 + [0.9] * 5)
            dfResults = calcularRendimientoTest(model, test_dir, analisis, paramsRed, thresholds)

            assert isinstance(dfResults, pd.DataFrame)
            assert not dfResults.empty
            assert dfResults.shape[1] == 16  # Número de columnas en el DataFrame

