# %%
import os
import sys
import shutil
import time
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import sqrt
from numpy import argmax
from datetime import datetime
from PIL import Image, ImageFile
from tensorflow import keras
from keras import layers, models, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #Para la AlexNet
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, auc, cohen_kappa_score, confusion_matrix
from keras.layers import BatchNormalization


from keras.callbacks import Callback
import keras.backend as K

class PrintLearningRate(Callback):
    def __init__(self, verbose=False):
        self.learning_rates = []
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        self.learning_rates.append(lr)
        if self.verbose:
            print(f'Learning rate en la epoch {epoch + 1}: {lr}')


class MultiEarlyStopping(Callback):
    def __init__(self, prim_monitor='val_auc', second_monitor='val_loss', prim_mode='auto', second_mode='auto', patience=1, restore_best_weights=False, start_from_epoch=0, baseline=None):
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


# %%
def copiar_imagenes(path_imagenes, df_imagenes, path_destino):
    for indice, fila in df_imagenes.iterrows():
        dataset = fila['Dataset']
        ruta_imagen = os.path.join(path_imagenes, dataset, fila['Nombre del archivo'])
        shutil.copy(ruta_imagen, path_destino)

# %%
def copiar_imagenes_hubu(path_imagenes, analisis, dir, dframe):
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    imagenes_hubu = dframe
    añadidas = [0] * len(clases)
    for label_id, clase in enumerate(clases):
        dir_clase = os.path.join(dir, clase)
        if not(os.path.exists(dir_clase)):
            os.makedirs(dir_clase)
        imagenes_de_la_clase = imagenes_hubu[imagenes_hubu[problema] == label_id]
        copiar_imagenes(path_imagenes, imagenes_de_la_clase, dir_clase)
        añadidas[label_id] += imagenes_de_la_clase.shape[0]

    posiciones_filas_hubu = imagenes_hubu.index.tolist()
    dframe.loc[posiciones_filas_hubu, 'sin usar'] = pd.NA

    return dframe, añadidas, imagenes_hubu.shape[0]

# %%
def copiar_imagenes_resto(path_imagenes, analisis, dir, dframe, num_local_images, conteo_local_por_clase):
    problema = analisis['objetivo']
    clases = analisis['clases']

    clase_mayoritaria = 1 # harcodeado para problema "Esta sano"
    clase_mayoritaria = 0 # harcodeado para problema "Esta sano"

    num_total_images = num_local_images*5
    
    añadir_para_equiparar = [max(conteo_local_por_clase)-numero for numero in conteo_local_por_clase]
    for label_id, clase in enumerate(clases):
        dir_clase = os.path.join(dir, clase)
        imagenes_resto = dframe[dframe['sin usar'] == 1]
        if añadir_para_equiparar[label_id] > 0:
            imagenes_resto_de_la_clase = imagenes_resto[imagenes_resto[problema] == label_id]
            filas_aleatorias = imagenes_resto_de_la_clase
            if filas_aleatorias.shape[0] < añadir_para_equiparar[label_id] and label_id != clase_mayoritaria:
                aun_por_añadir = añadir_para_equiparar[label_id] - filas_aleatorias.shape[0]
                imagenes_resto_ya_usadas = imagenes_resto[imagenes_resto['sin usar'] == 0]

                posiciones_a_resetear = imagenes_resto_ya_usadas.index.tolist()
                dframe.loc[posiciones_a_resetear, 'sin usar'] = 1
                if aun_por_añadir > 0:
                    filas_aleatorias_extra = imagenes_resto_ya_usadas.sample(n=min(aun_por_añadir, imagenes_resto_ya_usadas.shape[0]))
                filas_aleatorias         = pd.concat([filas_aleatorias, filas_aleatorias_extra], axis=0)
            elif filas_aleatorias.shape[0] > añadir_para_equiparar[label_id]:
                filas_aleatorias  = filas_aleatorias.sample(n=añadir_para_equiparar[label_id])
                
            copiar_imagenes(path_imagenes, filas_aleatorias, dir_clase)
            dframe.loc[filas_aleatorias.index.tolist(), 'sin usar'] = 0

    valor_base = num_total_images // len(clases)
    diferencia = num_total_images - (valor_base * len(clases))
    añadir_para_completar = [valor_base] * len(clases)
    for i in range(diferencia):
        añadir_para_completar[i] += 1
    
    for label_id, clase in enumerate(clases):
        dir_clase = os.path.join(dir, clase)
        
        añadir_para_completar[label_id] = añadir_para_completar[label_id] - conteo_local_por_clase[label_id] - añadir_para_equiparar[label_id]
        
        imagenes_resto = dframe
        imagenes_resto = imagenes_resto[imagenes_resto['sin usar'] == 1]
        if añadir_para_completar[label_id] > 0 and imagenes_resto.empty:
            imagenes_resto_de_la_clase = imagenes_resto[imagenes_resto[problema] == label_id]
            filas_aleatorias = imagenes_resto_de_la_clase
            if filas_aleatorias.shape[0] < añadir_para_completar[label_id] and label_id != clase_mayoritaria:
                aun_por_añadir = añadir_para_completar[label_id] - filas_aleatorias.shape[0]
                imagenes_resto_ya_usadas = imagenes_resto[imagenes_resto['sin usar'] == 0]

                posiciones_a_resetear = imagenes_resto_ya_usadas.index.tolist()
                dframe.loc[posiciones_a_resetear, 'sin usar'] = 1
                
                filas_aleatorias_extra   = imagenes_resto_ya_usadas.sample(n=min(aun_por_añadir, imagenes_resto_ya_usadas.shape[0]))
                filas_aleatorias         = pd.concat([filas_aleatorias, filas_aleatorias_extra], axis=0)
            elif filas_aleatorias.shape[0] > añadir_para_completar[label_id]:
                filas_aleatorias  = filas_aleatorias.sample(n=añadir_para_completar[label_id])
                
            copiar_imagenes(path_imagenes, filas_aleatorias, dir_clase)
            dframe.loc[filas_aleatorias.index.tolist(), 'sin usar'] = 0

    return dframe

# %%
def crear_dirstruc_train(path_imagenes, analisis, train_dir, train_dframe):
    estrategia_entreno = analisis['estrategia_entreno']
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    if not(os.path.exists(train_dir)):
        os.makedirs(train_dir)
    
    if 'sin usar' not in train_dframe.columns:
        train_dframe.insert(len(train_dframe.columns), 'sin usar', 1)
    else:
        train_dframe['sin usar'] = 1

    if estrategia_entreno == 'todo':
        for label_id, clase in enumerate(clases):
            dir_clase = os.path.join(train_dir, clase)
            if not(os.path.exists(dir_clase)):
                os.makedirs(dir_clase)
            imagenes_de_la_clase = train_dframe[train_dframe[problema] == label_id]
            copiar_imagenes(path_imagenes, imagenes_de_la_clase, dir_clase)
        
        train_dframe['sin usar'] = 0
    else:
        train_dframe, conteo_local_por_clase, num_local_images = copiar_imagenes_hubu(path_imagenes, analisis, train_dir, train_dframe)
        train_dframe = copiar_imagenes_resto(path_imagenes, analisis, train_dir, train_dframe, num_local_images, conteo_local_por_clase)
        
    return train_dframe

# %%
def crear_dirstruc_val(path_imagenes, analisis, val_dir, val_dframe):
    estrategia_entreno = analisis['estrategia_entreno']
    problema = analisis['objetivo']
    clases = analisis['clases']

    if not(os.path.exists(val_dir)):
        os.makedirs(val_dir)
    
    if 'sin usar' not in val_dframe.columns:
        val_dframe.insert(len(val_dframe.columns), 'sin usar', 1)
    else:
        val_dframe['sin usar'] = 1

    if estrategia_entreno == 'todo':
        for label_id, clase in enumerate(clases):
            dir_clase = os.path.join(val_dir, clase)
            if not(os.path.exists(dir_clase)):
                os.makedirs(dir_clase)
            imagenes_de_la_clase = val_dframe[val_dframe[problema] == label_id]
            copiar_imagenes(path_imagenes, imagenes_de_la_clase, dir_clase)
        
        val_dframe['sin usar'] = 1
    else:
        if val_dframe['sin usar'].isna().any():
            imagenes_ya_fijadas = val_dframe[val_dframe["sin usar"] != 1]
            for label_id, clase in enumerate(clases):
                dir_clase = os.path.join(val_dir, clase)
                if not(os.path.exists(dir_clase)):
                    os.makedirs(dir_clase)
                imagenes_de_la_clase = imagenes_ya_fijadas[imagenes_ya_fijadas[problema] == label_id]
                copiar_imagenes(path_imagenes, imagenes_de_la_clase, dir_clase)
        else:
            val_dframe, conteo_local_por_clase, num_local_images = copiar_imagenes_hubu(path_imagenes, analisis, val_dir, val_dframe)
            val_dframe = copiar_imagenes_resto(path_imagenes, analisis, val_dir, val_dframe, num_local_images, conteo_local_por_clase)
        
    return val_dframe

# %%
def crear_dirstruc(path_imagenes, analisis, dir, dframe):
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    if not(os.path.exists(dir)):
        os.makedirs(dir)
    
    for label_id, clase in enumerate(clases):
        dir_clase = os.path.join(dir, clase)
        if not(os.path.exists(dir_clase)):
            os.makedirs(dir_clase)
        imagenes_de_la_clase = dframe[dframe[problema] == label_id]
        copiar_imagenes(path_imagenes, imagenes_de_la_clase, dir_clase)

    return dframe

# %%
def data_structure_for_analysis(path, analisis, date_time, por_usar=None, iteracion = None):
    csv_file = analisis['fichero_csv']
    problema = analisis['objetivo']
    if(iteracion == None):
        iteracion = analisis['cviter']
    
    directorio_iteracion = os.path.join(path, date_time, f'iteracion{iteracion}')
    if os.path.exists(directorio_iteracion):
         shutil.rmtree(directorio_iteracion)
    os.makedirs(directorio_iteracion)
    
    idbox_testing    = iteracion
    idbox_validation = iteracion + 1
    if (idbox_validation > 5):
        idbox_validation = idbox_validation % 5
    idbox_training   = list(set(range(1,6)) - set([idbox_testing, idbox_validation]))

    if por_usar == None:
        # Carga el archivo CSV en un DataFrame de Pandas
        dataframe = pd.read_csv(csv_file)
        dataframe = dataframe[['Dataset', 'Nombre del archivo', problema, f'Caja {problema}']]
        dataframe = dataframe.dropna(subset=[problema])
    
        train_dframe = dataframe[dataframe[f'Caja {problema}'].isin(idbox_training)]
        train_dframe = train_dframe[['Dataset', 'Nombre del archivo', problema]]
        train_dframe = train_dframe.reset_index(drop=True)
        val_dframe   = dataframe[dataframe[f'Caja {problema}'] == idbox_validation]
        val_dframe   = val_dframe[['Dataset', 'Nombre del archivo', problema]]
        val_dframe   = val_dframe.reset_index(drop=True)
        test_dframe  = dataframe[dataframe[f'Caja {problema}'] == idbox_testing]
        test_dframe  = test_dframe[['Dataset', 'Nombre del archivo', problema]]
        test_dframe  = test_dframe.dropna()
        test_dframe  = test_dframe.reset_index(drop=True)
    else:
        train_dframe = por_usar['train']
        val_dframe   = por_usar['val']
        test_dframe  = por_usar['test']

    train_dir    = os.path.join(directorio_iteracion, 'training')
    val_dir      = os.path.join(directorio_iteracion, 'validation')
    test_dir      = os.path.join(directorio_iteracion, 'testing')
    
    directorios = {'train': train_dir,
                   'val': val_dir,
                   'test': test_dir}
    
    print(f"\nPasada numero: {iteracion}")
    print(f"Caja test: {idbox_testing}")
    print(f"Caja validación: {idbox_validation}")
    print(f"Caja training: {idbox_training}")
    print(f"Directorios: {directorios}")
    
    
    train_dframe = crear_dirstruc_train(os.path.abspath(paths['imagenes']), analisis, train_dir, train_dframe)
    val_dframe   = crear_dirstruc_val(os.path.abspath(paths['imagenes']), analisis, val_dir, val_dframe)
    test_dframe   = crear_dirstruc(os.path.abspath(paths['imagenes']), analisis, test_dir, test_dframe)
    
   
    print(f"\nTesting Humano " + str(test_dframe[test_dframe[problema] == 1].size))
    print(f"Testing No Humano " + str(test_dframe[test_dframe[problema] == 0].size))
    print(f"Validation Humano " + str(val_dframe[val_dframe[problema] == 1].size))
    print(f"Validation No Humano " + str(val_dframe[val_dframe[problema] == 0].size))
    print(f"Training Humano " + str(train_dframe[train_dframe[problema] == 1].size))
    print(f"Training No Humano " + str(train_dframe[train_dframe[problema] == 0].size))

    dframes = {'train': train_dframe,
               'val': val_dframe,
               'test': test_dframe}
    
    return directorios, dframes



# 
# %%
def data_structure_for_analysis_droneSAR(path, analisis, date_time, iteracion = None):
    csv_file = analisis['fichero_csv']
    problema = analisis['objetivo']
    if(iteracion == None):
        iteracion = analisis['cviter']
    
    directorio_iteracion = os.path.join(path, date_time, f'subimagenes')
    if os.path.exists(directorio_iteracion):
         shutil.rmtree(directorio_iteracion)
    os.makedirs(directorio_iteracion)
    
    idbox_testing    = iteracion
    idbox_validation = iteracion + 1
    if (idbox_validation > 5):
        idbox_validation = idbox_validation % 5
    idbox_training   = list(set(range(1,6)) - set([idbox_testing, idbox_validation]))

    # Carga el archivo CSV en un DataFrame de Pandas
    dataframe = pd.read_csv(csv_file)
    dataframe = dataframe[['Dataset', 'Nombre del archivo', problema, f'Caja {problema}']]
    dataframe = dataframe.dropna(subset=[problema])
    
    print(f"\tDataframe " + str(dataframe.size))    

    train_dframe = dataframe[dataframe[f'Caja {problema}'].isin(idbox_training)]
    train_dframe = train_dframe[['Dataset', 'Nombre del archivo', problema]]
    train_dframe = train_dframe.reset_index(drop=True)
    print(f"\t\ttrain_dframe " + str(train_dframe.size))    
    val_dframe   = dataframe[dataframe[f'Caja {problema}'] == idbox_validation]
    val_dframe   = val_dframe[['Dataset', 'Nombre del archivo', problema]]
    val_dframe   = val_dframe.reset_index(drop=True)
    print(f"\t\tval_dframe " + str(val_dframe.size))    
    test_dframe  = dataframe[dataframe[f'Caja {problema}'] == idbox_testing]
    test_dframe  = test_dframe[['Dataset', 'Nombre del archivo', problema]]
    test_dframe  = test_dframe.dropna()
    test_dframe  = test_dframe.reset_index(drop=True)
    print(f"\t\ttest_dframe " + str(test_dframe.size))    

    train_dir    = os.path.join(directorio_iteracion, 'training')
    val_dir      = os.path.join(directorio_iteracion, 'validation')
    test_dir      = os.path.join(directorio_iteracion, 'testing')
    
    directorios = {'train': train_dir,
                   'val': val_dir,
                   'test': test_dir}
    
    print(f"\nPasada numero: {iteracion}")
    print(f"Caja test: {idbox_testing}")
    print(f"Caja validación: {idbox_validation}")
    print(f"Caja training: {idbox_training}")
    
    
    train_dframe = crear_dirstruc_train(os.path.abspath(paths['imagenes']), analisis, train_dir, train_dframe)
    val_dframe   = crear_dirstruc_val(os.path.abspath(paths['imagenes']), analisis, val_dir, val_dframe)
    test_dframe   = crear_dirstruc(os.path.abspath(paths['imagenes']), analisis, test_dir, test_dframe)
    
    print(f"\nTesting Humano " + str(test_dframe[test_dframe[problema] == 1].size))
    print(f"Testing No Humano " + str(test_dframe[test_dframe[problema] == 0].size))
    print(f"Validation Humano " + str(val_dframe[val_dframe[problema] == 1].size))
    print(f"Validation No Humano " + str(val_dframe[val_dframe[problema] == 0].size))
    print(f"Training Humano " + str(train_dframe[train_dframe[problema] == 1].size))
    print(f"Training No Humano " + str(train_dframe[train_dframe[problema] == 0].size))


    dframes = {'train': train_dframe,
               'val': val_dframe,
               'test': test_dframe}
    
    return directorios, dframes


# %%
def configurar_metrica(metrica):
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

    #falta aun por incluir otras metricas!!

    return metricas, monitor, mode


# %%
def inicializar_callbacks(paths, date_time, analisis, params_red, monitor, mode):
    # Verificar si el directorio existe
    tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], date_time)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    res_dir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    printlr = PrintLearningRate()
    
    #es = EarlyStopping(monitor=monitor, mode=mode, patience=params_red['patience'], restore_best_weights=True, start_from_epoch=params_red['start_from_epoch'])
    mes = MultiEarlyStopping(prim_monitor=monitor, second_monitor='val_loss', prim_mode=mode, second_mode='min', patience=params_red['patience'], restore_best_weights=True, start_from_epoch=params_red['start_from_epoch'])
    filename = f"AlexNet{analisis['cviter']}.{date_time}."
    ckpt = ModelCheckpoint(filepath=os.path.join(tmp_dir, filename + 'weights.{epoch:03d}-{val_loss:.2f}.h5'),
                           monitor=monitor,
                           mode=mode,
                           save_weights_only=True,
                           save_best_only=True)
    csv_logger = CSVLogger(os.path.join(res_dir, filename + 'training.log'))

    return [printlr, mes, ckpt, csv_logger]


# %%
'''
def inicializar_alexnet(paths, date_time, analisis, params_red):
    model = models.Sequential()
    
    # Layer 1: Convolutional layer with 64 filters of size 11x11x3
    model.add(Conv2D(filters=64, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(340,340,3)))
    # Layer 2: Max pooling layer with pool size of 3x3
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
    model.add(Conv2D(filters=192, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # Layer 6: Fully connected layer with 4096 neurons
    model.add(layers.Flatten())

    num_capas_densas      = params_red['capas_densas']
    num_neuronas_por_capa = params_red['neuronas_por_capa']
    activacion_por_capa   = params_red['activacion_por_capa']
    dropout_por_capa      = params_red['dropout_por_capa']
    for capa_iesima in range(0, num_capas_densas):
        model.add(layers.Dense(units=num_neuronas_por_capa[capa_iesima], activation=activacion_por_capa[capa_iesima]))
        if dropout_por_capa[capa_iesima] > 0:
            model.add(layers.Dropout(dropout_por_capa[capa_iesima]))
    
    model.add(layers.Dense(1, activation='sigmoid'))
   
    #configurar metrica, compila optimizador, sin baseline
    metricas, monitor, mode = configurar_metrica(params_red['metrica'])
    #metrica a observar - 50-50
    # AUC-ROC

    loss = 'binary_crossentropy'
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params_red['learning_rate'], 
                                                                 decay_steps=params_red['decay_steps'],
                                                                 decay_rate=params_red['decay_rate'])
    
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metricas,
    )
    
    return model, monitor, mode
'''

def inicializar_alexnet(paths, date_time, analisis, params_red):
    model = models.Sequential()

    # Lista para almacenar información sobre cada capa
    arquitectura_red = []

    # Layer 1: Convolutional layer with 96 filters of size 16x16x3
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(340,340,3)))
    arquitectura_red.append({'Id de capa': 1, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 96, 'Descripción': 'Convolutional layer with 96 filters of size 11x11x3 and stride 4x4'})

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 2, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())

    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
    arquitectura_red.append({'Id de capa': 3, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 256, 'Descripción': 'Convolutional layer with 256 filters of size 11x11x3 and stride 1x1'})

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 4, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())

    
    # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitectura_red.append({'Id de capa': 5, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 384, 'Descripción': 'Convolutional layer with 384 filters of size 3x3x3 and stride 1x1'})

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitectura_red.append({'Id de capa': 6, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 384, 'Descripción': 'Convolutional layer with 384 filters of size 3x3x3 and stride 1x1'})

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    arquitectura_red.append({'Id de capa': 7, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 256, 'Descripción': 'Convolutional layer with 256 filters of size 3x3x3 and stride 1x1'})

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 8, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    model.add(BatchNormalization())

    
    # Layer 6: Fully connected layer with 4096 neurons
    model.add(layers.Flatten())
    arquitectura_red.append({'Id de capa': 9, 'Tipo de capa': 'Flatten', 'Neuronas de la capa': None, 'Descripción': 'Flatten layer to flatten the input for fully connected layer'})

    num_capas_densas = params_red['capas_densas']
    num_neuronas_por_capa = params_red['neuronas_por_capa']
    activacion_por_capa = params_red['activacion_por_capa']
    dropout_por_capa = params_red['dropout_por_capa']
    for capa_iesima in range(0, num_capas_densas):
        model.add(layers.Dense(units=num_neuronas_por_capa[capa_iesima], input_shape=(340,340,3), activation=activacion_por_capa[capa_iesima]))
        arquitectura_red.append({'Id de capa': 10+capa_iesima, 'Tipo de capa': 'Dense', 'Neuronas de la capa': num_neuronas_por_capa[capa_iesima], 'Descripción': f'Dense layer with {num_neuronas_por_capa[capa_iesima]} neurons and {activacion_por_capa[capa_iesima]} activation function'})
        if dropout_por_capa[capa_iesima] > 0:
            model.add(layers.Dropout(dropout_por_capa[capa_iesima]))
            arquitectura_red.append({'Id de capa': 10+num_capas_densas+capa_iesima, 'Tipo de capa': 'Dropout', 'Neuronas de la capa': None, 'Descripción': f'Dropout layer with dropout rate {dropout_por_capa[capa_iesima]}'})
            
        model.add(BatchNormalization())

    
    model.add(layers.Dense(1, activation='sigmoid'))
    arquitectura_red.append({'Id de capa': 10+num_capas_densas*2, 'Tipo de capa': 'Dense', 'Neuronas de la capa': 1, 'Descripción': 'Output layer with sigmoid activation function'})

    # Imprimir la arquitectura de la red
    print("Arquitectura de la red:")
    for capa_info in arquitectura_red:
        print(f"Capa {capa_info['Id de capa']}: {capa_info['Tipo de capa']}, Neuronas: {capa_info['Neuronas de la capa']}, Descripción: {capa_info['Descripción']}")

    # Configurar métrica, compilar optimizador
    metricas, monitor, mode = configurar_metrica(params_red['metrica'])
    loss = 'binary_crossentropy'
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params_red['learning_rate'], 
                                                                 decay_steps=params_red['decay_steps'],
                                                                 decay_rate=params_red['decay_rate'])
    
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metricas,
    )
    
    return model, monitor, mode


def inicializar_custom_red(paths, date_time, analisis, params_red):
    model = models.Sequential()

    # Lista para almacenar información sobre cada capa
    arquitectura_red = []

    # Layer 1: Convolutional layer with 64 filters of size 5x5x3
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(340,340,3)))
    arquitectura_red.append({'Id de capa': 1, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 64, 'Descripción': 'Convolutional layer with 64 filters of size 5x5x3 and stride 1x1'})

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 2, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    # Layer 3: Convolutional layer with 128 filters of size 3x3x64
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    arquitectura_red.append({'Id de capa': 3, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 128, 'Descripción': 'Convolutional layer with 128 filters of size 3x3x64 and stride 1x1'})

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 4, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    # Layer 4: Convolutional layer with 256 filters of size 3x3x128
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    arquitectura_red.append({'Id de capa': 5, 'Tipo de capa': 'Conv2D', 'Neuronas de la capa': 256, 'Descripción': 'Convolutional layer with 256 filters of size 3x3x128 and stride 1x1'})

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    arquitectura_red.append({'Id de capa': 6, 'Tipo de capa': 'MaxPooling2D', 'Neuronas de la capa': None, 'Descripción': 'MaxPooling layer with pool size 2x2 and stride 2x2'})

    # Layer 5: Flatten layer
    model.add(layers.Flatten())
    arquitectura_red.append({'Id de capa': 7, 'Tipo de capa': 'Flatten', 'Neuronas de la capa': None, 'Descripción': 'Flatten layer to flatten the input for fully connected layer'})

    num_capas_densas = params_red['capas_densas']
    num_neuronas_por_capa = params_red['neuronas_por_capa']
    activacion_por_capa = params_red['activacion_por_capa']
    dropout_por_capa = params_red['dropout_por_capa']
    for capa_iesima in range(0, num_capas_densas):
        model.add(layers.Dense(units=num_neuronas_por_capa[capa_iesima], activation=activacion_por_capa[capa_iesima]))
        arquitectura_red.append({'Id de capa': 8+capa_iesima, 'Tipo de capa': 'Dense', 'Neuronas de la capa': num_neuronas_por_capa[capa_iesima], 'Descripción': f'Dense layer with {num_neuronas_por_capa[capa_iesima]} neurons and {activacion_por_capa[capa_iesima]} activation function'})
        if dropout_por_capa[capa_iesima] > 0:
            model.add(layers.Dropout(dropout_por_capa[capa_iesima]))
            arquitectura_red.append({'Id de capa': 8+num_capas_densas+capa_iesima, 'Tipo de capa': 'Dropout', 'Neuronas de la capa': None, 'Descripción': f'Dropout layer with dropout rate {dropout_por_capa[capa_iesima]}'})
    
    model.add(layers.Dense(1, activation='sigmoid'))
    arquitectura_red.append({'Id de capa': 8+num_capas_densas*2, 'Tipo de capa': 'Dense', 'Neuronas de la capa': 1, 'Descripción': 'Output layer with sigmoid activation function'})

    # Imprimir la arquitectura de la red
    print("Arquitectura de la red:")
    for capa_info in arquitectura_red:
        print(f"Capa {capa_info['Id de capa']}: {capa_info['Tipo de capa']}, Neuronas: {capa_info['Neuronas de la capa']}, Descripción: {capa_info['Descripción']}")

    # Configurar métrica, compilar optimizador
    metricas, monitor, mode = configurar_metrica(params_red['metrica'])
    loss = 'binary_crossentropy'
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params_red['learning_rate'],
                                                                 decay_steps=params_red['decay_steps'],
                                                                 decay_rate=params_red['decay_rate'])

    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metricas,
    )

    return model, monitor, mode


def inicializar_custom_cnn(paths, date_time, analisis, params_red):
    model = models.Sequential()

    # Capa convolucional 1
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(340,340,3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa convolucional 2
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa convolucional 3
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa convolucional 4
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Capas densas
    model.add(layers.Flatten())
    
    num_capas_densas = params_red['capas_densas']
    num_neuronas_por_capa = params_red['neuronas_por_capa']
    activacion_por_capa = params_red['activacion_por_capa']
    dropout_por_capa = params_red['dropout_por_capa']
    for capa_iesima in range(0, num_capas_densas):
        model.add(layers.Dense(units=num_neuronas_por_capa[capa_iesima], activation=activacion_por_capa[capa_iesima]))
        if dropout_por_capa[capa_iesima] > 0:
            model.add(layers.Dropout(dropout_por_capa[capa_iesima]))
            
    model.add(layers.Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria

    # Configurar métrica, compilar optimizador
    metricas, monitor, mode = configurar_metrica(params_red['metrica'])
    loss = 'binary_crossentropy'
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params_red['learning_rate'], 
                                                                     decay_steps=params_red['decay_steps'],
                                                                 decay_rate=params_red['decay_rate'])

    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metricas,
    )

    return model, monitor, mode



# %%    
def inicializar_red(paths, date_time, analisis, params_red):
    modelo, monitor, mode = inicializar_alexnet(paths, date_time, analisis, params_red)
    #modelo, monitor, mode = inicializar_custom_red(paths, date_time, analisis, params_red)
    callbacks             = inicializar_callbacks(paths, date_time, analisis, params_red, monitor, mode)
    
    return modelo, callbacks    

# %%
def configurar_train_val_generators(analisis, params_red, train_dir, val_dir):

    modo = "binary"

    if analisis['data_augmentation'] == 0:
        train_datagen = ImageDataGenerator(
                rescale=1.0 / 255)
    else:
        train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2)
    
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(340, 340),  # Tamaño de entrada de AlexNet
            color_mode=params_red['color'],
            batch_size=params_red['train_batch_size'],
            class_mode=modo,
            classes=analisis['clases'],
            shuffle=True)

    if analisis['data_augmentation'] == 0:
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255)
    else:
        val_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2)
        
    val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(340, 340),  # Tamaño de entrada de AlexNet
            color_mode=params_red['color'],
            batch_size=params_red['val_batch_size'],
            class_mode=modo,
            classes=analisis['clases'],
            shuffle=False)

    return train_generator, val_generator


# %%
def calcular_pesos_por_clase(train_generator):
    path_training = train_generator.directory
    clases = list(train_generator.class_indices.keys())

    directorios_base = []
    for clase in clases:
        directorios_base.append(os.path.join(path_training, clase))

    # Crear una lista para almacenar el número de imágenes por cada clase
    num_imagenes_por_clase = []
    
    # Iterar sobre los subdirectorios
    for subdirectorio in directorios_base:
        archivos = os.listdir(subdirectorio)
        num_imagenes = len(archivos)
        num_imagenes_por_clase.append(num_imagenes)

        # Print para ver la ruta donde se encuentran las imágenes de cada directorio
        print(f"Directorio: {subdirectorio}")
        print(f"Número de imágenes en el directorio: {num_imagenes}")

    # Print para ver el número total de imágenes
    total_imagenes = sum(num_imagenes_por_clase)
    print(f"Número total de imágenes: {total_imagenes}")

    # Cálculo de los pesos por clase
    class_weights = [total_imagenes / numero for numero in num_imagenes_por_clase]
    # Convertir la lista de pesos por clase a un diccionario
    class_weights = {indice: valor for indice, valor in enumerate(class_weights)}

    # Print para ver los pesos asignados a cada clase
    print("Pesos asignados a cada clase:")
    for clase, peso in zip(clases, class_weights.values()):
        print(f"Clase: {clase}, Peso: {peso}")

    return class_weights
 
# %%

# %%
def entrenamiento_simple(modelo, callbacks, train_generator, val_generator, params_red):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class_weights = calcular_pesos_por_clase(train_generator)
    
    # Guarda el tiempo de inicio
    inicio = time.time()
    
    history = modelo.fit(
        train_generator,
        epochs = params_red['epochs'],
        callbacks = callbacks,
        validation_data = val_generator,
        class_weight = class_weights
    )
    
    # Guarda el tiempo de finalización
    fin = time.time()
    # Calculo del tiempo de entrenamiento en segundos
    duracion_segundos = fin - inicio
    # Convierte la duración a minutos
    duracion_minutos = duracion_segundos / 60

    historico = pd.DataFrame(history.history)
    historico.insert(0, 'lr', callbacks[0].learning_rates)

    return modelo, historico, duracion_minutos


def obtener_mejor_del_historico(historico, row_id):
    mejor = historico.iloc[[row_id-1]]
    
    return mejor


# In[ ]:
def pasada_uno(paths, date_time, analisis, params_red, modelo, callbacks, train_generator, val_generator):
    tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], date_time)

    print(f"PASADA 1 - CVITER {analisis['cviter']}")
    
    modelo, historico, _ = entrenamiento_simple(modelo, callbacks, train_generator, val_generator, params_red)
    
    historico.insert(0, 'pasada', 1)
    best_performance = obtener_mejor_del_historico(historico, callbacks[1].best_epoch)
    print(f"Mejor {params_red['metrica']} en validacion hasta ahora:")
    print(best_performance)
    
    archivos_tmp = os.listdir(tmp_dir)
    for archivo in archivos_tmp:
        os.remove(os.path.join(tmp_dir, archivo))

    #guardar modelo en disco
    fichero = os.path.join(tmp_dir, f"{params_red['modelo_base']}.{analisis['cviter']}.pasada1.modelo.h5")
    modelo.save(fichero)

    return modelo, historico, best_performance


# In[ ]:
def promediar_modelos(modelo, nuevo_modelo):
    capas_modelo1 = modelo.layers
    capas_dense_modelo1 = [capa for capa in capas_modelo1 if isinstance(capa, layers.Dense)]
    capas_modelo2 = nuevo_modelo.layers
    capas_dense_modelo2 = [capa for capa in capas_modelo2 if isinstance(capa, layers.Dense)]

    pesos_promediados = []
    # Iterar sobre todas las capas de los modelos
    for capa_modelo1, capa_modelo2 in zip(capas_dense_modelo1, capas_dense_modelo2):
        # Obtener los pesos de ambas capas
        pesos_modelo1 = capa_modelo1.get_weights()
        pesos_modelo2 = capa_modelo2.get_weights()
            
        # Calcular el promedio de los pesos
        pesos_promediados_capa = [(w1 + w2) / 2 for w1, w2 in zip(pesos_modelo1, pesos_modelo2)]
            
        # Agregar los pesos promediados a la lista
        pesos_promediados.append(pesos_promediados_capa)
            

    # Establecer los pesos promediados en uno de los modelos
    for capa, pesos_promediados_capa in zip(capas_dense_modelo1, pesos_promediados):
        capa.set_weights(pesos_promediados_capa)
        
    return modelo


# In[ ]:
def pasada_iesima(num_pasada, historico, best_performance, paths, date_time, analisis, params_red, modelo, dirs_entrenamiento, total_sin_usar):
    tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], date_time)

    print()
    print(f"PASADA {num_pasada} - CVITER {analisis['cviter']}")
    print(f"Quedan sin usar: {total_sin_usar}")
    
    train_generator, val_generator = configurar_train_val_generators(analisis, params_red, dirs_entrenamiento['train'], dirs_entrenamiento['val'])
    nuevo_modelo, callbacks = inicializar_red(paths, date_time, analisis, params_red, modelo)
    
    nuevo_modelo, nuevo_historico, _ = entrenamiento_simple(nuevo_modelo, callbacks, train_generator, val_generator, params_red)
    
    nuevo_historico.insert(0, 'pasada', num_pasada)
    new_performance = obtener_mejor_del_historico(nuevo_historico, callbacks[1].best_epoch)
    historico = pd.concat([historico, nuevo_historico], ignore_index=True)
        
    nueva_metrica = 1 - new_performance[f"val_{params_red['metrica']}"].iloc[0]  #harcodeado para metricas a maximizar
    mejor_metrica = 1 - best_performance[f"val_{params_red['metrica']}"].iloc[0] #harcodeado para metricas a maximizar
    nuevo_loss = new_performance["val_loss"].iloc[0]
    mejor_loss = best_performance["val_loss"].iloc[0]

    if nueva_metrica == 0:
        nueva_metrica = 0.001
    elif nuevo_loss == 0:
        nuevo_loss = 0.001
    
    nuevo_resultado = nueva_metrica*nuevo_loss

    if mejor_metrica == 0:
        mejor_metrica = 0.001
    elif mejor_loss == 0:
        mejor_loss = 0.001
    
    mejor_resultado = mejor_metrica*mejor_loss
    
    if  nuevo_resultado <= mejor_resultado:
        print("Se ha mejorado las metricas que se estan optimizando")
        best_performance = new_performance
        modelo = nuevo_modelo
    #elif nueva_metrica > mejor_metrica or nuevo_loss < mejor_loss:
    #    print("Se mejora solo una de los dos metricas que se esta optimizando, promediamos el mejor modelo hasta ahora con el actual entrenado")
    #    modelo = promediar_modelos(modelo, nuevo_modelo)
        
    print(f"Mejor {params_red['metrica']} en validacion hasta ahora:")
    print(best_performance)

    archivos_tmp = os.listdir(tmp_dir)
    for archivo in archivos_tmp:
        if "pasada" not in archivo:
            os.remove(os.path.join(tmp_dir, archivo))

    #guardar modelo en disco
    fichero = os.path.join(tmp_dir, f"{params_red['modelo_base']}.{analisis['cviter']}.pasada{num_pasada}.modelo.h5")
    modelo.save(fichero)
    os.remove(os.path.join(tmp_dir, f"{params_red['modelo_base']}.{analisis['cviter']}.pasada{num_pasada-1}.modelo.h5"))

    return modelo, historico, best_performance
# %%

# %%
def metricas_binclass(y_real, y_pred, threshold=0.5):
    '''Funcion que calcula las metricas de rendimiento habituales para un clasificador binario.
    
    Parametros
    ----------
    y_real : np.array de 1 dimension
        Etiquetas reales, cada una representada por un número natural, de cada una de las muestras.
    y_pred : np.array de 1 dimension
        Etiquetas predichas, representadas por una probabilidad entre 0 y 1, de cada una de las muestras.
        
    Retorno: una lista con las metricas [N, % 0's, % 1's, TN, FN, TP, FP, Accuracy, NPV, PPV, TNR, TPR, F1Score, AU-PR, AU-ROC]
    '''
    
    porcentaje_unos = np.mean(y_real) * 100
    porcentaje_ceros = 100 - porcentaje_unos

    #umbral = obtener_mejor_threshold(y_real, y_pred, thres_criteria)
    umbral = threshold
    
    # Binarizar el vector de probabilidades
    y_pred_binario = np.where(y_pred >= umbral, 1, 0)

    # Calcular la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred_binario).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    npv = tn / (tn + fn + np.finfo(float).eps) #negative predictive value
    ppv = tp / (tp + fp + np.finfo(float).eps) #tambien conocido como precision
    tnr = tn / (tn + fp + np.finfo(float).eps) #tambien conocida como especificidad, la proporción de verdaderos negativos sobre el total de instancias negativas reales
    tpr = tp / (tp + fn + np.finfo(float).eps) #tambien conocida como sensibilidad o recall, la proporción de verdaderos positivos sobre el total de instancias positivas reales
    f1 = (2 * ppv * tpr) / (ppv + tpr  + np.finfo(float).eps)

    precision, recall, thresholds = precision_recall_curve(y_real, y_pred)
    au_pr  = auc(recall, precision)
    au_roc = roc_auc_score(y_real, y_pred)

    metricas = [sum([tn,fp,fn,tp]), porcentaje_ceros, porcentaje_unos, umbral, tn, fn, tp, fp, accuracy, npv, ppv, tnr, tpr, f1, au_pr, au_roc]
    
    return metricas, y_pred_binario

# %%
def calcular_rendimiento_test(modelo, test_dir, analisis, params_red):
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )
    
    columnas = ['N', "% 0's", "% 1's", 'Umbral', 'TN', 'FN', 'TP', 'FP', 'Accuracy', 'NPV', 'PPV', 'TNR', 'TPR', 'F1Score', 'AU-PR', 'AU-ROC']
    df_results = pd.DataFrame(columns=columnas)
    
    # Ajusta el tamaño de la imagen para AlexNet
    target_size = (340, 340)  # Tamaño de entrada de AlexNet
    
    # Genera el generador de imágenes de prueba
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode=params_red['color'],
        batch_size=params_red['test_batch_size'],
        class_mode='binary',  # Siempre binaria para AlexNet
        classes = analisis['clases'],
        shuffle=False)

    # Obtiene las etiquetas verdaderas de prueba
    y_test_true = test_generator.labels
    # Obtiene las predicciones del modelo para las imágenes de prueba
    y_test_prob = modelo.predict(test_generator)
    
    # Calcula las métricas de rendimiento
    resultado = metricas_binclass(y_test_true, y_test_prob)

    # Insert data into DataFrame
    df_results.loc[len(df_results)] = resultado[0]
    
    return df_results



# %% [markdown]
# # PROGRAMA PRINCIPAL

# %% [markdown]
# Capturamos el nombre del script y cargamos el archivo de configuracion con los diferentes parametros para ejecutar el analisis.
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S") # guardamos la hora actual para usarlo junto a otra informacion como id del analisis

#nombre_script = sys.argv[0]
nombre_script = 'entrenamiento_cv_alexnet.py'

with open('./parametros.yaml', 'r') as archivo_config:
    configuracion = yaml.safe_load(archivo_config)

configuracion['analisis']['script'] = nombre_script

analisis     = configuracion["analisis"]
paths        = configuracion["paths"]
params_red   = configuracion["red_neuronal"]

# aqui, idealmente, habria que invocar a una funcion que comprobara que la parametrizacion recibida es correcta para el buen funcionamiento del script y,
# de no serlo, informar e interrumpir la ejecución del resto del script
# dicha funcion aun falta por implementar!

# %% [markdown]
# Vemos si se quiere ejecutar el analisis en una GPU concreta.
if analisis['id_gpu'] >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[analisis['id_gpu']], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[analisis['id_gpu']], True)
    print(f"ELEGIDA LA GPU Nº {analisis['id_gpu']}")


# %% [markdown]
# Creamos la estructura de directorios para la iteracion de validacion cruzada en cuestion a partir del fichero CSV con todas las imagenes y la particion de cajas de validacion cruzada.
# - El contenido de la funcion data_structure_for_analysis se puede modificar y adaptar en base a las necesidades del problema, acorde a la informacion que haya en el fichero_csv correspondiente
# - Lo imprescindible es que cree los directorios de train, validation y test para entrenar y testar el modelo.

directorios, imagenes_por_usar = data_structure_for_analysis_droneSAR(os.path.abspath(paths['datos_entreno']), analisis, date_time)


# Inicializamos la arquitectura de la red a entrenar, junto al optimizador, metricas a monitorizar y diferentes callbacks de control del entrenamiento.

# In[ ]:


modelo, callbacks = inicializar_red(paths, date_time, analisis, params_red)


# Configuramos los ImageGenerators para ir cogiendo y procesando imagenes de training/validacion por batches acorde a la entrada que espera la red a entrenar.

# In[ ]:


train_generator, val_generator = configurar_train_val_generators(analisis, params_red, directorios['train'], directorios['val'])


# Entrenamos el modelo con las configuraciones realizadas previamente.

# In[ ]:


res_dir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'], date_time)

if analisis['estrategia_entreno'] == 'todo':
    modelo, _, tiempo = entrenamiento_simple(modelo, callbacks, train_generator, val_generator, params_red)
else:
    modelo, historico, tiempo = entrenamiento_iterativo(paths, date_time, analisis, modelo, callbacks, train_generator, val_generator, params_red, imagenes_por_usar)
    historico.to_csv(os.path.join(res_dir, f"{params_red['modelo_base']}.{analisis['cviter']}.training.log"), index=False)


# Guardamos el modelo entrenado para poder usarlo posteriormente en inferencias; Calculamos las metricas de rendimiento en la(s) carpeta(s) de test y guardamos en un fichero csv los resultados obtenidos.

# In[ ]:


fichero = os.path.join(res_dir, f"AlexNet.{analisis['cviter']}.modelo.h5")
modelo.save(fichero)

resultados_test = calcular_rendimiento_test(modelo, directorios['test'], analisis, params_red)
resultados_test.insert(0, 'Iteracion', analisis['cviter'])
resultados_test.insert(0, 'Problema', analisis['objetivo'])
resultados_test.insert(len(resultados_test.columns), 'Tiempo (mins)', tiempo)
    
print(resultados_test)
resultados_test.to_csv(os.path.join(res_dir, f"AlexNet.{analisis['cviter']}.resultados.csv"), index=False)

# In[ ]:
# Guardamos la configuracion de parametros con la que se ejecuto este script.
fichero = os.path.join(res_dir, f"AlexNet.{analisis['cviter']}.params_config.yaml")

with open(fichero, 'w') as archivo:
    yaml.dump(configuracion, archivo)


# In[ ]:
directorio_iteracion = os.path.join(os.path.abspath(paths['datos_entreno']), date_time)
shutil.rmtree(directorio_iteracion)

tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'], date_time)
shutil.rmtree(tmp_dir)






'''

def entrenamiento_cruzado(paths, date_time, analisis, modelo, callbacks, train_generator, val_generator, params_red, dirs_entrenamiento):
    tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'])
    filename = f"{analisis['cviter']}.{date_time}.weights"
    problema = analisis['objetivo']
    iteracion = analisis['cviter']
    numberofboxes = analisis['numberofboxes']
    
    # Guarda el tiempo de inicio
    inicio = time.time()
    print("PASADA ", iteracion)
    modelo, _ = entrenamiento_simple(modelo, callbacks, train_generator, val_generator, params_red)
    
    for iter in range(0,numberofboxes-1):
        
        # Guardamos el modelo entrenado para poder usarlo posteriormente en inferencias.
        res_dir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
        fichero = os.path.join(res_dir, f"{analisis['cviter']}.{date_time}.modelo.keras")
        modelo.save(fichero)

        # Calculamos las metricas de rendimiento en la(s) carpeta(s) de test y guardamos en un fichero csv los resultados obtenidos.
        resultados_test = calcular_rendimiento_test(modelo, directorios['test'], analisis, params_red)

        resultados_test.insert(0, 'Iteracion', analisis['cviter'])
        resultados_test.insert(0, 'Problema', analisis['objetivo'])
        
        # Guarda el tiempo de finalización
        fin = time.time()
        # Calculo del tiempo de entrenamiento en segundos
        duracion_segundos = fin - inicio
        # Convierte la duración a minutos
        duracion_minutos = duracion_segundos / 60
        
        resultados_test.insert(len(resultados_test.columns), 'Tiempo (mins)', duracion_minutos)

        inicio = time.time()

        print(resultados_test)
        output_path=os.path.join(res_dir, f"resultados.csv")
        resultados_test.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))

        # Guardamos la configuracion de parametros con la que se ejecuto este script.
        fichero = os.path.join(res_dir, f"{analisis['cviter']}.{date_time}.params_config.yaml")

        with open(fichero, 'w') as archivo:
            yaml.dump(configuracion, archivo)

        currentIter = ((iteracion + iter)%numberofboxes)+1
        print("PASADA ", currentIter)
        
        # Inicializamos la arquitectura de la red a entrenar, junto al optimizador, metricas a monitorizar y diferentes callbacks de control del entrenamiento.
        modelo, callbacks = inicializar_red(paths, date_time, analisis, params_red)

        # Configuramos los ImageGenerators para ir cogiendo y procesando imagenes de training/validacion por batches acorde a la entrada que espera la red a entrenar.
        dirs_entrenamiento, imagenes_por_usar = data_structure_for_analysis(os.path.abspath(paths['datos_entreno']), analisis, currentIter)
        archivos_tmp = os.listdir(tmp_dir)
        for archivo in archivos_tmp:
            if archivo.startswith(filename):
                os.remove(os.path.join(tmp_dir, archivo))
        
        train_generator, val_generator = configurar_train_val_generators(analisis, params_red, dirs_entrenamiento['train'], dirs_entrenamiento['val'])
        modelo, _ = entrenamiento_simple(modelo, callbacks, train_generator, val_generator, params_red)
        
    # Guarda el tiempo de finalización
    fin = time.time()
    # Calculo del tiempo de entrenamiento en segundos
    duracion_segundos = fin - inicio
    # Convierte la duración a minutos
    duracion_minutos = duracion_segundos / 60
    
    directorio_iteracion = os.path.join(os.path.abspath(paths['datos_entreno']), f'subimagenes')
    if os.path.exists(directorio_iteracion):
         shutil.rmtree(directorio_iteracion)

    return modelo, duracion_minutos




def entrenamiento_iterativo(paths, date_time, analisis, modelo, callbacks, train_generator, val_generator, params_red, imagenes_por_usar):
    
    problema = analisis['objetivo']
    
    # Guarda el tiempo de inicio
    inicio = time.time()

    modelo, historico, best_performance = pasada_uno(paths, date_time, analisis, params_red, modelo, callbacks, train_generator, val_generator)
    
    imagenes_sin_usar = imagenes_por_usar['train'][[problema,'sin usar']]
    imagenes_sin_usar = imagenes_sin_usar.dropna()
    imagenes_sin_usar = imagenes_sin_usar[imagenes_sin_usar[problema] == 1] #Hardcodeado, hay que poner la clase mayoritaria
    total_sin_usar = sum(imagenes_sin_usar['sin usar'])
    num_pasada = 2
    while total_sin_usar > 0:
        keras.backend.clear_session()

        dirs_entrenamiento, imagenes_por_usar = data_structure_for_analysis(os.path.abspath(paths['datos_entreno']), analisis, date_time, imagenes_por_usar)
        modelo, historico, best_performance = pasada_iesima(num_pasada, historico, best_performance, paths, date_time, analisis, params_red, modelo, dirs_entrenamiento, total_sin_usar)
        
        imagenes_sin_usar = imagenes_por_usar['train'][[problema,'sin usar']]
        imagenes_sin_usar = imagenes_sin_usar.dropna()
        imagenes_sin_usar = imagenes_sin_usar[imagenes_sin_usar[problema] == 1] #Hardcodeado, hay que poner la clase mayoritaria
        total_sin_usar = sum(imagenes_sin_usar['sin usar'])
        
        num_pasada += 1

    print(f"Mejor {params_red['metrica']} en validacion final:")
    print(best_performance)
    
    # Guarda el tiempo de finalización
    fin = time.time()
    # Calculo del tiempo de entrenamiento en segundos
    duracion_segundos = fin - inicio
    # Convierte la duración a minutos
    duracion_minutos = duracion_segundos / 60

    return modelo, historico, duracion_minutos



    
    
tmp_dir = os.path.join(os.path.abspath(paths['temporal']), analisis['objetivo'])
problema = analisis['objetivo']
iteracion = analisis['cviter']
currentIter = iteracion
numberofboxes = analisis['numberofboxes']        
for iter in range(0, numberofboxes):
    print("PASADA ", currentIter)
    configuracion['analisis']['cviter'] = currentIter
    # Actualizamos la iteración en el yaml
    with open('./parametros.yaml', 'w') as archivo_config:
        yaml.safe_dump(configuracion, archivo_config, default_flow_style=False)
    # Limpiamos los datos previos y cargamos la nueva estructura de la iteración
    archivos_tmp = os.listdir(tmp_dir)
    for archivo in archivos_tmp:
        os.remove(os.path.join(tmp_dir, archivo))
    directorios, imagenes_por_usar = data_structure_for_analysis_droneSAR(os.path.abspath(paths['datos_entreno']), analisis)

    # Inicializamos la red
    modelo, callbacks = inicializar_red(paths, date_time, analisis, params_red)
    # Configuramos los ImageGenerators para ir cogiendo y procesando imagenes de training/validacion por batches acorde a la entrada que espera la red a entrenar.
    train_generator, val_generator = configurar_train_val_generators(analisis, params_red, directorios['train'], directorios['val'])

    # Entrenamos el modelo con las configuraciones realizadas previamente.
    modelo, historico, tiempo = entrenamiento_iterativo(paths, date_time, analisis, modelo, callbacks, train_generator, val_generator, params_red, directorios, imagenes_por_usar)

    # Guardamos el modelo entrenado para poder usarlo posteriormente en inferencias.
    res_dir = os.path.join(os.path.abspath(paths['resultados']), analisis['objetivo'])
    fichero = os.path.join(res_dir, f"{currentIter}.{date_time}.modelo.h5")
    modelo.save(fichero)
    # Calculamos las metricas de rendimiento en la(s) carpeta(s) de test y guardamos en un fichero csv los resultados obtenidos.
    resultados_test = calcular_rendimiento_test(modelo, directorios['test'], analisis, params_red)
    resultados_test.insert(0, 'Iteracion', currentIter)
    resultados_test.insert(0, 'Problema', analisis['objetivo'])
    resultados_test.insert(len(resultados_test.columns), 'Tiempo (mins)', tiempo)
    print(resultados_test)
    resultados_test.to_csv(os.path.join(res_dir, f"{currentIter}.{date_time}.resultados.csv"), index=False)
    # Guardamos la configuracion de parametros con la que se ejecuto este script.
    fichero = os.path.join(res_dir, f"{currentIter}.{date_time}.params_config.yaml")
    with open(fichero, 'w') as archivo:
        yaml.dump(configuracion, archivo)

    currentIter = ((iteracion + iter)%numberofboxes)+1
    
elif(analisis['estrategia_entreno'] == 'todo'):
    print("a")

    
# Volvemos a dejar la iteración inicial en el yaml
configuracion['analisis']['cviter'] = iteracion
        
# Actualizamos la iteración en el yaml
with open('./parametros.yaml', 'w') as archivo_config:
    yaml.safe_dump(configuracion, archivo_config, default_flow_style=False)
'''
