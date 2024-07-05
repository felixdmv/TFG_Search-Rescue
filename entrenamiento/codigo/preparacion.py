# Preparation.py
import os
import shutil
import yaml
import pandas as pd
import tensorflow as tf
from settingsEntrenamiento import PATH_PARAMETROS
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def copiarImagenes(pathImagenes, dfImagenes, pathDestino):
    for indice, fila in dfImagenes.iterrows():
        dataset = fila['Dataset']
        rutaImagen = os.path.join(pathImagenes, dataset, dataset, fila['Nombre del archivo'])
        shutil.copy(rutaImagen, pathDestino)


def copiarImagenesDf(pathImagenes, analisis, dir, dframe):
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    imagenesHubu = dframe
    añadidas = [0] * len(clases)
    for labelId, clase in enumerate(clases):
        dirClase = os.path.join(dir, clase)
        if not(os.path.exists(dirClase)):
            os.makedirs(dirClase)
        imagenesDeLaClase = imagenesHubu[imagenesHubu[problema] == labelId]
        copiarImagenes(pathImagenes, imagenesDeLaClase, dirClase)
        añadidas[labelId] += imagenesDeLaClase.shape[0]

    posicionesFilasHubu = imagenesHubu.index.tolist()
    dframe.loc[posicionesFilasHubu, 'sin usar'] = pd.NA

    return dframe, añadidas, imagenesHubu.shape[0]


def copiarImagenesResto(pathImagenes, analisis, dir, dframe, numLocalImages, conteoLocalPorClase):
    problema = analisis['objetivo']
    clases = analisis['clases']

    claseMayoritaria = 1 # Harcodeado para problema "Esta sano"
    claseMayoritaria = 0 # Harcodeado para problema "Esta sano"

    numTotalImages = numLocalImages * 5
    
    añadirParaEquiparar = [max(conteoLocalPorClase) - numero for numero in conteoLocalPorClase]
    for labelId, clase in enumerate(clases):
        dirClase = os.path.join(dir, clase)
        imagenesResto = dframe[dframe['sin usar'] == 1]
        if añadirParaEquiparar[labelId] > 0:
            imagenesRestoDeLaClase = imagenesResto[imagenesResto[problema] == labelId]
            filasAleatorias = imagenesRestoDeLaClase
            if filasAleatorias.shape[0] < añadirParaEquiparar[labelId] and labelId != claseMayoritaria:
                aunPorAñadir = añadirParaEquiparar[labelId] - filasAleatorias.shape[0]
                imagenesRestoYaUsadas = imagenesResto[imagenesResto['sin usar'] == 0]

                posicionesAResetear = imagenesRestoYaUsadas.index.tolist()
                dframe.loc[posicionesAResetear, 'sin usar'] = 1
                if aunPorAñadir > 0:
                    filasAleatoriasExtra = imagenesRestoYaUsadas.sample(n=min(aunPorAñadir, imagenesRestoYaUsadas.shape[0]))
                filasAleatorias = pd.concat([filasAleatorias, filasAleatoriasExtra], axis=0)
            elif filasAleatorias.shape[0] > añadirParaEquiparar[labelId]:
                filasAleatorias = filasAleatorias.sample(n=añadirParaEquiparar[labelId])
                
            copiarImagenes(pathImagenes, filasAleatorias, dirClase)
            dframe.loc[filasAleatorias.index.tolist(), 'sin usar'] = 0

    valorBase = numTotalImages // len(clases)
    diferencia = numTotalImages - (valorBase * len(clases))
    añadirParaCompletar = [valorBase] * len(clases)
    for i in range(diferencia):
        añadirParaCompletar[i] += 1
    
    for labelId, clase in enumerate(clases):
        dirClase = os.path.join(dir, clase)
        
        añadirParaCompletar[labelId] = añadirParaCompletar[labelId] - conteoLocalPorClase[labelId] - añadirParaEquiparar[labelId]
        
        imagenesResto = dframe
        imagenesResto = imagenesResto[imagenesResto['sin usar'] == 1]
        if añadirParaCompletar[labelId] > 0 and imagenesResto.empty:
            imagenesRestoDeLaClase = imagenesResto[imagenesResto[problema] == labelId]
            filasAleatorias = imagenesRestoDeLaClase
            if filasAleatorias.shape[0] < añadirParaCompletar[labelId] and labelId != claseMayoritaria:
                aunPorAñadir = añadirParaCompletar[labelId] - filasAleatorias.shape[0]
                imagenesRestoYaUsadas = imagenesResto[imagenesResto['sin usar'] == 0]

                posicionesAResetear = imagenesRestoYaUsadas.index.tolist()
                dframe.loc[posicionesAResetear, 'sin usar'] = 1
                
                filasAleatoriasExtra = imagenesRestoYaUsadas.sample(n=min(aunPorAñadir, imagenesRestoYaUsadas.shape[0]))
                filasAleatorias = pd.concat([filasAleatorias, filasAleatoriasExtra], axis=0)
            elif filasAleatorias.shape[0] > añadirParaCompletar[labelId]:
                filasAleatorias = filasAleatorias.sample(n=añadirParaCompletar[labelId])
                
            copiarImagenes(pathImagenes, filasAleatorias, dirClase)
            dframe.loc[filasAleatorias.index.tolist(), 'sin usar'] = 0

    return dframe


def crearDirstrucTrain(pathImagenes, analisis, trainDir, trainDframe):
    estrategiaEntreno = analisis['estrategiaEntreno']
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    if not(os.path.exists(trainDir)):
        os.makedirs(trainDir)
    
    if 'sin usar' not in trainDframe.columns:
        trainDframe.insert(len(trainDframe.columns), 'sin usar', 1)
    else:
        trainDframe['sin usar'] = 1

    if estrategiaEntreno == 'todo':
        for labelId, clase in enumerate(clases):
            dirClase = os.path.join(trainDir, clase)
            if not(os.path.exists(dirClase)):
                os.makedirs(dirClase)
            imagenesDeLaClase = trainDframe[trainDframe[problema] == labelId]
            copiarImagenes(pathImagenes, imagenesDeLaClase, dirClase)
        
        trainDframe['sin usar'] = 0
    else:
        trainDframe, conteoLocalPorClase, numLocalImages = copiarImagenesDf(pathImagenes, analisis, trainDir, trainDframe)
        trainDframe = copiarImagenesResto(pathImagenes, analisis, trainDir, trainDframe, numLocalImages, conteoLocalPorClase)
        
    return trainDframe


def crearDirstrucVal(pathImagenes, analisis, valDir, valDframe):
    estrategiaEntreno = analisis['estrategiaEntreno']
    problema = analisis['objetivo']
    clases = analisis['clases']

    if not(os.path.exists(valDir)):
        os.makedirs(valDir)
    
    if 'sin usar' not in valDframe.columns:
        valDframe.insert(len(valDframe.columns), 'sin usar', 1)
    else:
        valDframe['sin usar'] = 1

    if estrategiaEntreno == 'todo':
        for labelId, clase in enumerate(clases):
            dirClase = os.path.join(valDir, clase)
            if not(os.path.exists(dirClase)):
                os.makedirs(dirClase)
            imagenesDeLaClase = valDframe[valDframe[problema] == labelId]
            copiarImagenes(pathImagenes, imagenesDeLaClase, dirClase)
        
        valDframe['sin usar'] = 1
    else:
        if valDframe['sin usar'].isna().any():
            imagenesYaFijadas = valDframe[valDframe["sin usar"] != 1]
            for labelId, clase in enumerate(clases):
                dirClase = os.path.join(valDir, clase)
                if not(os.path.exists(dirClase)):
                    os.makedirs(dirClase)
                imagenesDeLaClase = imagenesYaFijadas[imagenesYaFijadas[problema] == labelId]
                copiarImagenes(pathImagenes, imagenesDeLaClase, dirClase)
        else:
            valDframe, conteoLocalPorClase, numLocalImages = copiarImagenesDf(pathImagenes, analisis, valDir, valDframe)
            valDframe = copiarImagenesResto(pathImagenes, analisis, valDir, valDframe, numLocalImages, conteoLocalPorClase)
        
    return valDframe


def crearDirstruc(pathImagenes, analisis, dir, dframe):
    problema = analisis['objetivo']
    clases = analisis['clases']
    
    if not(os.path.exists(dir)):
        os.makedirs(dir)
    
    for labelId, clase in enumerate(clases):
        dirClase = os.path.join(dir, clase)
        if not(os.path.exists(dirClase)):
            os.makedirs(dirClase)
        imagenesDeLaClase = dframe[dframe[problema] == labelId]
        copiarImagenes(pathImagenes, imagenesDeLaClase, dirClase)

    return dframe


def dataStructureForAnalysisDroneSAR(path, analisis, dateTime, iteracion = None):
    csvFile = analisis['ficheroCsv']
    problema = analisis['objetivo']
    if(iteracion == None):
        iteracion = analisis['cviter']
    
    directorioIteracion = os.path.join(path, dateTime, f'subimagenes')
    if os.path.exists(directorioIteracion):
         shutil.rmtree(directorioIteracion)
    os.makedirs(directorioIteracion)
    
    idboxTesting    = iteracion
    idboxValidation = iteracion + 1
    if (idboxValidation > 5):
        idboxValidation = idboxValidation % 5
    idboxTraining   = list(set(range(1,6)) - set([idboxTesting, idboxValidation]))

    # Carga el archivo CSV en un DataFrame de Pandas
    dataframe = pd.read_csv(csvFile)
    dataframe = dataframe[['Dataset', 'Nombre del archivo', problema, f'Caja {problema}']]
    dataframe = dataframe.dropna(subset=[problema])
    
    print(f"\tDataframe " + str(dataframe.size))    

    trainDframe = dataframe[dataframe[f'Caja {problema}'].isin(idboxTraining)]
    trainDframe = trainDframe[['Dataset', 'Nombre del archivo', problema]]
    trainDframe = trainDframe.reset_index(drop=True)
    print(f"\t\ttrain_dframe " + str(trainDframe.size))    
    valDframe   = dataframe[dataframe[f'Caja {problema}'] == idboxValidation]
    valDframe   = valDframe[['Dataset', 'Nombre del archivo', problema]]
    valDframe   = valDframe.reset_index(drop=True)
    print(f"\t\tval_dframe " + str(valDframe.size))    
    testDframe  = dataframe[dataframe[f'Caja {problema}'] == idboxTesting]
    testDframe  = testDframe[['Dataset', 'Nombre del archivo', problema]]
    testDframe  = testDframe.dropna()
    testDframe  = testDframe.reset_index(drop=True)
    print(f"\t\ttest_dframe " + str(testDframe.size))    

    trainDir    = os.path.join(directorioIteracion, 'training')
    valDir      = os.path.join(directorioIteracion, 'validation')
    testDir      = os.path.join(directorioIteracion, 'testing')
    
    directorios = {'train': trainDir,
                   'val': valDir,
                   'test': testDir}
    
    print(f"\nPasada numero: {iteracion}")
    print(f"Caja test: {idboxTesting}")
    print(f"Caja validación: {idboxValidation}")
    print(f"Caja training: {idboxTraining}")
    
    #with open('./parametros.yaml', 'r') as archivoConfig:
    
    with open(str(PATH_PARAMETROS), 'r') as archivoConfig:    
        configuracion = yaml.safe_load(archivoConfig)

    paths = configuracion["paths"]
    
    trainDframe = crearDirstrucTrain(os.path.abspath(paths['imagenes']), analisis, trainDir, trainDframe)
    valDframe   = crearDirstrucVal(os.path.abspath(paths['imagenes']), analisis, valDir, valDframe)
    testDframe   = crearDirstruc(os.path.abspath(paths['imagenes']), analisis, testDir, testDframe)
    
    print(f"\nTesting Humano " + str(testDframe[testDframe[problema] == 1].size))
    print(f"Testing No Humano " + str(testDframe[testDframe[problema] == 0].size))
    print(f"Validation Humano " + str(valDframe[valDframe[problema] == 1].size))
    print(f"Validation No Humano " + str(valDframe[valDframe[problema] == 0].size))
    print(f"Training Humano " + str(trainDframe[trainDframe[problema] == 1].size))
    print(f"Training No Humano " + str(trainDframe[trainDframe[problema] == 0].size))


    dframes = {'train': trainDframe,
               'val': valDframe,
               'test': testDframe}
    
    return directorios, dframes

def configurarTrainValGenerators(analisis, paramsRed, trainDir, valDir):

    modo = "binary"

    if analisis['dataAugmentation'] == 0:
        trainDatagen = ImageDataGenerator(
                rescale=1.0 / 255)
    else:
        trainDatagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2)
    
    trainGenerator = trainDatagen.flow_from_directory(
            trainDir,
            target_size=(340, 340),  # Tamaño de entrada de AlexNet
            color_mode=paramsRed['color'],
            batch_size=paramsRed['trainBatchSize'],
            class_mode=modo,
            classes=analisis['clases'],
            shuffle=True)

    if analisis['dataAugmentation'] == 0:
        valDatagen = ImageDataGenerator(
            rescale=1.0 / 255)
    else:
        valDatagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2)
        
    valGenerator = valDatagen.flow_from_directory(
            valDir,
            target_size=(340, 340),  # Tamaño de entrada de AlexNet
            color_mode=paramsRed['color'],
            batch_size=paramsRed['valBatchSize'],
            class_mode=modo,
            classes=analisis['clases'],
            shuffle=False)

    return trainGenerator, valGenerator
