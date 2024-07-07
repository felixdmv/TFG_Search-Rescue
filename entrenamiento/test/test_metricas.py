import pytest
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from unittest.mock import MagicMock, patch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from metricas import configurarMetrica, metricasBinclass, calcularRendimientoTest


@pytest.fixture(scope='module')
def setup_data():
    """
    Creates fictitious data for testing purposes.

    Returns:
        yReal (numpy.ndarray): Array of real values.
        yPred (numpy.ndarray): Array of predicted values.
    """
    yReal = np.array([0, 0, 1, 1])
    yPred = np.array([0.1, 0.4, 0.35, 0.8])
    return yReal, yPred


def test_configurarMetrica():
    """
    Test function for the configurarMetrica function.
    """
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
    """
    Test function for metricasBinclass.

    Args:
        setup_data: A tuple containing the real labels (yReal) and predicted labels (yPred).

    Returns:
        None
    """
    yReal, yPred = setup_data
    metricas, yPredBinario = metricasBinclass(yReal, yPred, threshold=0.5)
    
    assert len(metricas) == 16
    assert isinstance(metricas, list)
    assert isinstance(yPredBinario, np.ndarray)


def test_metricasBinclass_values(setup_data):
    """
    Test function to verify the values of various metrics for binary classification.

    Parameters:
    - setup_data: A tuple containing the real values (yReal) and predicted values (yPred).

    Returns:
    None
    """
    yReal, yPred = setup_data
    metricas, yPredBinario = metricasBinclass(yReal, yPred, threshold=0.5)
    
    # Verifica los valores de las métricas
    assert metricas[4] == 2  # TN
    assert metricas[5] == 1  # FN
    assert metricas[6] == 1  # TP
    assert metricas[7] == 0  # FP
    assert np.isclose(metricas[8], 0.75, atol=1e-2)  # Accuracy
    assert np.isclose(metricas[9], 0.66, atol=1e-2)  # NPV
    assert np.isclose(metricas[10], 1, atol=1e-2)  # PPV
    assert np.isclose(metricas[11], 1, atol=1e-2)  # TNR
    assert np.isclose(metricas[12], 0.5, atol=1e-2)  # TPR
    assert np.isclose(metricas[13], 0.66, atol=1e-2)  # F1Score
    assert np.isclose(metricas[14], 0.79, atol=1e-2)  # AU-PR
    assert np.isclose(metricas[15], 0.75, atol=1e-2)  # AU-ROC


@pytest.fixture(scope='module')
def setup_model():
    """
    Creates a simple model for testing purposes.

    Returns:
        model (Sequential): The compiled model with an input shape of (10,) and a single Dense layer with 1 unit.
    """
    model = models.Sequential([layers.Dense(1, input_shape=(10,))])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def test_calcularRendimientoTest(setup_model, tmpdir):
    """
    Test function for calculating performance metrics on test data.

    Args:
        setup_model: The setup model object.
        tmpdir: The temporary directory for storing test images.

    Returns:
        None
    """
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