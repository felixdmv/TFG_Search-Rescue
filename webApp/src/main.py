import settings
import utils.entradaSalida as es
import utils.graficosImagenes as gi
import prediccion as pred
import widgets as wd
from transitions import Machine
import gps
import streamlit as st

class AppStateMachine:
    states = ['excepcion', 'inicio', 'cargandoModelo', 'hayModelo', 'hayImagenes', 'predicciones', 'imagenesProcesadas']

    def __init__(self):
        self.machine = Machine(model=self, states=AppStateMachine.states, initial='inicio')
        self.machine.add_transition('reset', '*', 'inicio')
        self.machine.add_transition('excepcion', '*', 'excepcion')
        self.machine.add_transition('cargaModelo', ['inicio', 'hayModelo', 'hayImagenes'], 'cargandoModelo')
        self.machine.add_transition('cargandoModelo_hayModelo', 'cargandoModelo', 'hayModelo')
        self.machine.add_transition('cargandoModelo_hayImagenes', 'cargandoModelo', 'hayImagenes')
        self.machine.add_transition('hayModelo_hayImagenes', 'hayModelo', 'hayImagenes')
        self.machine.add_transition('hayImagenes_predicciones', 'hayImagenes', 'predicciones')
        self.machine.add_transition('predicciones_imagenesProcesadas', 'predicciones', 'imagenesProcesadas')
        self.machine.add_transition('imagenesProcesadas_hayImagenes', 'imagenesProcesadas', 'hayImagenes')
        self.machine.add_transition('hayImagenes_hayModelo', 'hayImagenes', 'hayModelo')


def inicializaEstadosYFSM():
    st.session_state['fsm'] = AppStateMachine()
    parametros = cargaParametrosConfiguracion(settings.PATH_PARAMETROS)
    if parametros == None:
        st.session_state['mensajeExcepcion'] = "Error cargando el fichero de configuración de parámetros."
        st.session_state.fsm.excepcion()
        st.rerun()
    try:
        st.session_state['enlacesModelos'] = cargaEnlacesModelos()  # Desde secrets.toml
    except Exception as e:
        st.session_state['mensajeExcepcion'] = "Error cargando los enlaces de los modelos."
        st.session_state.fsm.excepcion()
        st.rerun()
    try:
        st.session_state['nombresColeccionesYPath'] = cargaColecciones(settings.PATH_COLECCIONESIMAGENES)
    except Exception as e:
        st.session_state['mensajeExcepcion'] = "Error cargando los enlaces de las colecciones de imágenes."
        st.session_state.fsm.excepcion()
        st.rerun()

    st.session_state['keyUploader'] = 0
    st.session_state['listaImagenes'] = []
    st.session_state['listaImagenesLocal'] = None # True si local, False si servidor
    st.session_state['nombreModelo'] = None
    st.session_state['nombreColeccion'] = None
    st.session_state['indiceColeccion'] = None

    st.session_state['sliderUmbralPrediccion'] = parametros['sliderUmbralPrediccion']
    st.session_state['numberInputNumColumnas'] = parametros['numberInputNumColumnas']
    st.session_state['imageExtensions'] = parametros['imagenes']['imageExtensions']
    st.session_state['overlap'] = parametros['imagenes']['overlap']
    st.session_state['margins'] = parametros['imagenes']['margins']
    st.session_state['sizeSubimagenes'] = parametros['imagenes']['size']
    st.session_state['numColumnasTablaDurantePrediccion'] = parametros['numColumnasTablaDurantePrediccion']
    st.session_state['anchoLineaRectangulos'] = parametros['anchoLineaRectangulos']
    st.session_state['colorRectangulos'] = parametros['colorRectangulos']


@st.cache_data(ttl=3600)  # 1 hora de persistencia
def cargaParametrosConfiguracion(ficheroConfiguracion):
    '''
    Es un envoltorio de la función cargaParametrosConfiguracionYAML de entradasSalidas.py para cachear los resultados.
    '''
    return es.cargaParametrosConfiguracionYAML(ficheroConfiguracion)


@st.cache_resource(ttl=3600)  # 1 hora de persistencia
def cargaEnlacesModelos():
    """
    Carga los enlaces de los modelos desde la configuración de secretos y los devuelve en un diccionario.

    Returns:
        dict: Un diccionario que contiene los enlaces de los modelos, donde las claves son los nombres de los modelos
        y los valores son las URLs de los enlaces.
    """
    enlacesModelos = {}
    for enlace in st.secrets['modelos']:
        enlacesModelos[enlace[0]] = enlace[1]
    return enlacesModelos

@st.cache_data(ttl=3600, show_spinner=False)  # Para que no muestre el spinner por defecto
def cargaModeloH5(nombreModelo, urlModelo):
    """
    Carga un modelo a partir de un nombre h5 y su url y devuelve el modelo cargado.

    Args:
        nombreModelo (str): El nombre del modelo.
        urlModelo (str): La URL del archivo del modelo.

    Returns:
        El modelo h5 cargado.

    """
    ficheroModelo = nombreModelo # + '.h5'
    
    es.cargaArchivoDrive(urlModelo, ficheroModelo) # Carga localmente el modelo desde Drive
    return pred.cargaModelo(ficheroModelo)
    
    

st.cache_data(ttl=3600)  # 1 hora de persistencia
def cargaColecciones(path):
    """
    Load collections from the given path and return a dictionary with collection names as keys and their corresponding paths as values.

    Args:
        path (Path): The path to the directory containing the collections.

    Returns:
        dict: A dictionary mapping collection names to their paths.
    """
    coleccionesPath = [f for f in path.iterdir()]
    nombresColeccionesYPath = {}
    for path in coleccionesPath:
        parametros = cargaParametrosConfiguracion(path.joinpath(settings.DESCRIPCION_COLECCION))
        nombresColeccionesYPath[parametros['nombre']] = path
    return nombresColeccionesYPath


@st.cache_data(ttl=3600)  # 1 hora de persistencia
def creaRectangulos(tamImOriginal, tamSubimagen, solapamiento, margenes):
    """
    Es un envoltorio de la función creaListaRectangulosConIndices de graficosImagenes.py para cachear los resultados.

    Creates a list of rectangles with indices based on the given parameters.

    Args:
        tamImOriginal (tuple): The size of the original image.
        tamSubimagen (tuple): The size of the subimage.
        solapamiento (int): The amount of overlap between subimages.
        margenes (int): The margin size around the subimages.

    Returns:
        list: A list of rectangles. Los índices no se usan en esta aplicación
    """
    listaRectangulosConIndices = gi.creaListaRectangulosConIndices(tamImOriginal, tamSubimagen, solapamiento, margenes)
    return [rectangulo for rectangulo, _ in listaRectangulosConIndices]


def creaImagenConHumanos(imagen, predicciones, listaRectangulos, umbral, ancho=4, color='red'):
    """
    Crea una imagen donde las subimágnes con humanos se resaltan dibundo un rectángulo

    Args:
        imagen (PIL.Image.Image): The image on which to draw the rectangles.
        predicciones (list): A list of predictions.
        listaRectangulos (list): A list of rectangles.
        umbral (float): The threshold value.
        ancho (int): The width of the lines used to draw the rectangles. Defaults to 4.
        color (str): The color of the lines used to draw the rectangles. Defaults to 'red'.

    Returns:
        tuple: Una tupla con los rectángulos con humanos y una booleana que indica si en la imagen se detectó un humano
    """
    listaRectangulosConHumanos = []
    hayHumanos = False
    for i, prediccion in enumerate(predicciones):
        if prediccion + 0.005 >= umbral:  # 0.005 to round up and make the slider at 1.00 with a prediction of 0.998, for example, to be considered as a human found
            listaRectangulosConHumanos.append(listaRectangulos[i])
            hayHumanos = True
    imagenConHumanos = gi.dibujaRectangulos(imagen, listaRectangulosConHumanos, ancho, color)
    return imagenConHumanos, hayHumanos
  

def inicio():
    st.session_state['nombreModelo'] = wd.selectboxSeleccionModelo(list(st.session_state.enlacesModelos.keys()))
    if st.session_state['nombreModelo'] is not None:
        st.session_state.fsm.cargaModelo()  
        st.rerun()

def seleccionModelo():
    nombreModelo = wd.selectboxSeleccionModelo(list(st.session_state.enlacesModelos.keys()))
    if nombreModelo != st.session_state.nombreModelo:
        if nombreModelo is None:
            st.session_state.fsm.reset()
        else:
            st.session_state.nombreModelo = nombreModelo
            st.session_state.fsm.cargaModelo()
        st.rerun()


def cargandoModelo():
    with st.spinner(f'Cargando modelo {st.session_state.nombreModelo}...'):
        urlModelo = st.session_state.enlacesModelos[st.session_state.nombreModelo]
        # paso str(st.session_state.nombreModelo) porque parece que no es cacheable un
        # argumento que es una variable st.session_state
        st.session_state['modelo'] = cargaModeloH5(str(st.session_state.nombreModelo), urlModelo)
    if len(st.session_state['listaImagenes']):
        st.session_state.fsm.cargandoModelo_hayImagenes()
    else:
        st.session_state.fsm.cargandoModelo_hayModelo()
    st.rerun()


def seleccionImagenes():
    wd.formularioParametros()
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            wd.cargaImagenesLocal()
        with col2:
            wd.cargaImagenesServidor()
    if st.session_state.listaImagenes:
        if st.session_state.fsm.state == 'hayModelo':
            st.session_state.fsm.hayModelo_hayImagenes()
            st.rerun()
    else:
        if st.session_state.fsm.state == 'hayImagenes':
            st.session_state.fsm.hayImagenes_hayModelo()
            st.rerun()
    

def muestraImagenes():
    if st.session_state.nombreColeccion is None:
        textoBoton = f"Procesar imágenes seleccionadas localmente "
    else:
        textoBoton = f"Procesar imágenes de la colección {st.session_state.nombreColeccion}"
    if st.button(textoBoton):
        st.session_state.fsm.hayImagenes_predicciones()
        st.rerun()
    st.session_state.numberInputNumColumnas[2] = st.number_input("Selecciona número de columnas de visualización",
                                                                    *st.session_state.numberInputNumColumnas)
    with st.spinner('Creando tabla de imágenes...'):
        wd.creaTablaImagenes(st.session_state['listaImagenes'], st.session_state.numberInputNumColumnas[2])  
    

def procesandoImagenes():
    if st.session_state.nombreColeccion is None:
        st.write('Colección de imágenes seleccionadas localmente')
    else:
        st.write(f'Colección de imágenes de prueba "{st.session_state.nombreColeccion}"')
    
    st.write(f'Procesando con Modelo {st.session_state["nombreModelo"]}')
    with st.spinner('Creando tabla de imágenes...'):
        wd.creaTablaImagenes(st.session_state['listaImagenes'], 8) 
    sizeSubimagen = st.session_state.sizeSubimagenes
    overlap = st.session_state.overlap
    margins = st.session_state.margins
    umbral = st.session_state.sliderUmbralPrediccion[2]
    numImagenes = len(st.session_state.listaImagenes)
    st.session_state['listaImagenesProcesadas'] = []
    st.session_state['resultados'] = 'Nombre, Latitud, Longitud\n'
    for i, imagen in enumerate(st.session_state.listaImagenes):
        
        listaRectangulos = creaRectangulos(imagen.size, sizeSubimagen, overlap, margins)
        with st.spinner(f'Procesando imagen {i+1} de {numImagenes}'):
            try:
                st.session_state['predicciones'] = pred.predice(st.session_state['modelo'], imagen, listaRectangulos)
            except Exception as e:
                st.session_state['resultados'] += f"Error procesando la imagen {st.session_state.nombresImagenes[i]}"
            else:
                imagenProcesada, hayHumano = creaImagenConHumanos(imagen, st.session_state['predicciones'], listaRectangulos, umbral,
                                                                  st.session_state.anchoLineaRectangulos, st.session_state.colorRectangulos)
                st.session_state['listaImagenesProcesadas'].append(imagenProcesada)
                if hayHumano:
                    latitud, longitud = gps.extraeLatitudLongitud(imagen)
                    st.session_state['resultados'] += f'{st.session_state.nombresImagenes[i]}, {latitud}, {longitud}\n'

    st.session_state.fsm.predicciones_imagenesProcesadas()
    st.rerun()


def imagenesProcesadas():
    if st.session_state.nombreColeccion is None:
        st.write('Colección de imágenes seleccionadas localmente')
    else:
        st.write(f'Colección de imágenes de prueba "{st.session_state.nombreColeccion}"')
    st.write(f'Imágenes procesadas con Modelo {st.session_state["nombreModelo"]}')
    st.session_state.numberInputNumColumnas[2] = st.number_input("Selecciona número de columnas de visualización", 
                                                           *st.session_state.numberInputNumColumnas)

    with st.spinner('Creando tabla de imágenes...'):
        wd.creaTablaImagenes(st.session_state['listaImagenesProcesadas'], st.session_state.numberInputNumColumnas[2])     
        
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Volver a página principal"):
                st.session_state.fsm.imagenesProcesadas_hayImagenes()
                st.rerun()
        with col2:
            st.download_button("Descargar csv", st.session_state['resultados'], mime="text/csv")

def main():
    # Se hace str() al PATH porque st.logo() espera un string no un objeto Path
    st.logo(str(settings.PATH_LOGO), link=settings.URL_LOGO)
    st.subheader("Detección de Personas en Operaciones de Búsqueda y Rescate con UAS")

    if 'fsm' not in st.session_state:
        inicializaEstadosYFSM()

    fsm = st.session_state.fsm
    if fsm.state == 'excepcion':
        st.error(st.session_state.mensajeExcepcion)
        st.stop()
    elif fsm.state == 'inicio':
        inicio()
    elif fsm.state == 'cargandoModelo':
        cargandoModelo()
    elif fsm.state == 'hayModelo':
        seleccionModelo()    
        seleccionImagenes()
    elif fsm.state == 'hayImagenes':
        seleccionModelo()    
        seleccionImagenes()
        muestraImagenes()
    elif fsm.state == 'predicciones':
        procesandoImagenes()
    elif fsm.state == 'imagenesProcesadas':
        imagenesProcesadas()

    
if __name__ == '__main__':
    main()  