# neural_network.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #Para la AlexNet
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, auc, cohen_kappa_score, confusion_matrix
from keras.layers import BatchNormalization
from keras.callbacks import Callback
import keras.backend as K
from metricas import configurarMetrica


class PrintLearningRate(Callback):
    """
    Callback class to print the learning rate at the end of each epoch.

    Attributes:
        learning_rates (list): List to store the learning rates.
        verbose (bool): Flag to indicate whether to print the learning rate or not.

    Methods:
        on_epoch_end(epoch, logs=None): Method called at the end of each epoch to print the learning rate.
    """

    def __init__(self, verbose=False):
        """
        Initialize the PrintLearningRate object.

        Args:
            verbose (bool, optional): Flag to indicate whether to print the learning rate or not. Defaults to False.
        """
        self.learning_rates = []
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called at the end of each epoch to print the learning rate.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing the training metrics for the current epoch. Defaults to None.
        """
        lr = float(K.get_value(self.model.optimizer.lr))
        self.learning_rates.append(lr)
        if self.verbose:
            print(f'Learning rate in epoch {epoch + 1}: {lr}')


class MultiEarlyStopping(Callback):
    """
    Custom callback for implementing multi-monitor early stopping during model training.

    Args:
        prim_monitor (str): The primary metric to monitor for improvement. Default is 'val_auc'.
        second_monitor (str): The secondary metric to monitor for improvement. Default is 'val_loss'.
        prim_mode (str): The mode of the primary metric. Default is 'auto'.
        second_mode (str): The mode of the secondary metric. Default is 'auto'.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 1.
        restore_best_weights (bool): Whether to restore the weights of the best model found during training. Default is False.
        start_from_epoch (int): The epoch number from which to start monitoring for improvement. Default is 0.
        baseline: Baseline value for the monitored metrics. Default is None.
    """

    def __init__(self, prim_monitor='val_auc', second_monitor='val_loss', prim_mode='auto', second_mode='auto', patience=1, restore_best_weights=False, start_from_epoch=0, baseline=None):
        """
        Initializes a MultiEarlyStopping object.

        Parameters:
        - prim_monitor (str): The primary metric to monitor. Defaults to 'val_auc'.
        - second_monitor (str): The secondary metric to monitor. Defaults to 'val_loss'.
        - prim_mode (str): The mode for the primary metric. Defaults to 'auto'.
        - second_mode (str): The mode for the secondary metric. Defaults to 'auto'.
        - patience (int): The number of epochs to wait before stopping training. Defaults to 1.
        - restore_best_weights (bool): Whether to restore the weights from the epoch with the best performance. Defaults to False.
        - start_from_epoch (int): The epoch number to start counting patience from. Defaults to 0.
        - baseline (float): The baseline value for the monitored metric. Defaults to None.
        """
        super(MultiEarlyStopping, self).__init__()
        self.prim_monitor = prim_monitor
        self.second_monitor = second_monitor
        self.mode1 = prim_mode
        self.mode2 = second_mode
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.baseline = baseline
        self.wait = 0
        self.best_epoch = 0
        self.monitor_op = np.less
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch during training.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the metrics of the model on the validation set.

        Returns:
            None
        """
        if epoch < self.start_from_epoch:
            return
        
        current1 = logs.get(self.prim_monitor)
        current2 = logs.get(self.second_monitor)
        
        if current1 is None or current2 is None:
            return
        
        if self.mode1 == "max":
            current1 = 1-current1
        if self.mode2 == "max":
            current2 = 1-current2

        if current1 == 0:
            current1 = 0.001
        elif current2 == 0:
            current2 = 0.001
        
        current_metric = current1 * current2
            
        if self.monitor_op(current_metric, self.best):
            print(f'\nNuevo mejor modelo: {current_metric}\n')
            self.best_epoch = epoch+1
            self.best = current_metric
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'\nEpoch {epoch+1}: Early stopping due to no improvement in {self.prim_monitor} * {self.second_monitor}.')
                if self.restore_best_weights:
                    print(f'Restoring model weights from the end of the best epoch #{self.best_epoch}; {self.best}\n')
                    self.model.set_weights(self.best_weights)
                                  
                    
def inicializarCallbacks(paths, dateTime, analisis, paramsRed, monitor, mode):
    """
    Initialize the callbacks for the neural network training.

    Args:
        paths (dict): A dictionary containing the paths for the temporary and results directories.
        dateTime (str): The current date and time.
        analisis (dict): A dictionary containing the analysis parameters.
        paramsRed (dict): A dictionary containing the neural network parameters.
        monitor (str): The metric to monitor.
        mode (str): The mode for the monitored metric.

    Returns:
        list: A list of callbacks to be used during training.

    """
    # Verificar si el directorio existe
    tmpDir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], dateTime)
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)

    res_dir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    printlr = PrintLearningRate()
    
    #es = EarlyStopping(monitor=monitor, mode=mode, patience=paramsRed['patience'], restore_best_weights=True, start_from_epoch=paramsRed['start_from_epoch'])
    mes = MultiEarlyStopping(prim_monitor=monitor, second_monitor='val_loss', prim_mode=mode, second_mode='min', patience=paramsRed['patience'], restore_best_weights=True, start_from_epoch=paramsRed['startFromEpoch'])
    filename = f"AlexNet{analisis['cviter']}.{dateTime}."
    ckpt = ModelCheckpoint(filepath=os.path.join(tmpDir, filename + 'weights.h5'),
                           monitor=monitor,
                           mode=mode,
                           save_weights_only=True,
                           save_best_only=True)
    csvLogger = CSVLogger(os.path.join(res_dir, filename + 'training.log'))

    return [printlr, mes, ckpt, csvLogger]


def inicializarAlexnet(paths, dateTime, analisis, paramsRed):
    """
    Initializes the AlexNet model with the specified parameters.

    Args:
        paths (list): List of file paths.
        dateTime (str): Date and time information.
        analisis (str): Analysis information.
        paramsRed (dict): Dictionary of network parameters.

    Returns:
        model (Sequential): Initialized AlexNet model.
        monitor (str): Metric to monitor during training.
        mode (str): Mode of the metric to monitor.

    Raises:
        None

    """
    model = models.Sequential()

    # Lista para almacenar información sobre cada capa
    arquitecturaRed = []

    # Layer 1: Convolutional layer with 96 filters of size 16x16x3
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(340,340,3)))
    arquitecturaRed.append({'Id de capa': 1, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 96, 'Descripción': 'Convolutional layer with 96 filters of size 11x11x3 and stride 4x4'})

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitecturaRed.append({'Id de capa': 2, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())
 
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
    arquitecturaRed.append({'Id de capa': 3, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 256, 'Descripción': 'Convolutional layer with 256 filters of size 11x11x3 and stride 1x1'})

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitecturaRed.append({'Id de capa': 4, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())
    
    # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitecturaRed.append({'Id de capa': 5, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 384, 'Descripción': 'Convolutional layer with 384 filters of size 3x3x3 and stride 1x1'})

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitecturaRed.append({'Id de capa': 6, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 384, 'Descripción': 'Convolutional layer with 384 filters of size 3x3x3 and stride 1x1'})

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitecturaRed.append({'Id de capa': 7, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 256, 'Descripción': 'Convolutional layer with 256 filters of size 3x3x3 and stride 1x1'})

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitecturaRed.append({'Id de capa': 8, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())

    # Layer 6: Fully connected layer with 4096 neurons
    model.add(layers.Flatten())
    arquitecturaRed.append({'Id de capa': 9, 'Tipo de capa': 'Flatten', 'Neuronas de la capa': None, 'Descripción': 'Flatten layer to flatten the input for fully connected layer'})

    numCapasDensas = paramsRed['capasDensas']
    numNeuronasPorCapa = paramsRed['neuronasPorCapa']
    activacionPorCapa = paramsRed['activacionPorCapa']
    dropoutPorCapa = paramsRed['dropoutPorCapa']
    for capaIesima in range(0, numCapasDensas):
        model.add(layers.Dense(units=numNeuronasPorCapa[capaIesima], input_shape=(340,340,3), activation=activacionPorCapa[capaIesima]))
        arquitecturaRed.append({'Id de capa': 10+capaIesima, 'Tipo de capa': 'Dense', 'Neuronas de la capa': numNeuronasPorCapa[capaIesima], 'Descripción': f'Dense layer with {numNeuronasPorCapa[capaIesima]} neurons and {activacionPorCapa[capaIesima]} activation function'})
        if dropoutPorCapa[capaIesima] > 0:
            model.add(layers.Dropout(dropoutPorCapa[capaIesima]))
            arquitecturaRed.append({'Id de capa': 10+numCapasDensas+capaIesima, 'Tipo de capa': 'Dropout', 'Neuronas de la capa': None, 'Descripción': f'Dropout layer with dropout rate {dropoutPorCapa[capaIesima]}'})
            
        model.add(BatchNormalization())

    model.add(layers.Dense(1, activation='sigmoid'))
    arquitecturaRed.append({'Id de capa': 10+numCapasDensas*2, 'Tipo de capa': 'Dense', 'Neuronas de la capa': 1, 'Descripción': 'Output layer with sigmoid activation function'})

    # Imprimir la arquitectura de la red
    print("Arquitectura de la red:")
    for capaInfo in arquitecturaRed:
        print(f"Capa {capaInfo['Id de capa']}: {capaInfo['Tipo de capa']}, Neuronas: {capaInfo['Neuronas de la capa']}, Descripción: {capaInfo['Descripción']}")

    # Configurar métrica, compilar optimizador
    metricas, monitor, mode = configurarMetrica(paramsRed['metrica'])
    loss = 'binary_crossentropy'
    lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(paramsRed['learningRate'], 
                                                                 decay_steps=paramsRed['decaySteps'],
                                                                 decay_rate=paramsRed['decayRate'])
    
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lrSchedule),
        loss = loss,
        metrics = metricas,
    )
    
    return model, monitor, mode


def inicializarRed(paths, dateTime, analisis, paramsRed):
    """
    Initializes the neural network.

    Args:
        paths (list): List of file paths.
        dateTime (str): Date and time information.
        analisis (str): Analysis information.
        paramsRed (dict): Dictionary of neural network parameters.

    Returns:
        tuple: A tuple containing the initialized model and callbacks.
    """
    modelo, monitor, mode = inicializarAlexnet(paths, dateTime, analisis, paramsRed)
    callbacks             = inicializarCallbacks(paths, dateTime, analisis, paramsRed, monitor, mode)
    
    return modelo, callbacks