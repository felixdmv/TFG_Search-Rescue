import settings
import utils.entradaSalida as es
import utils.graficosImagenes as gi
import utils.utilidadesDirectorios as ud
import prediccion as pred
import widgets as wd
import streamlit as st

def inicializaEstadosSesion():
    """
    Initializes the session states.

    This function sets the initial values for various session states used in the application.
    """
    st.session_state['estadosInicializados'] = True # Indica si los estados de la sesión han sido inicializados
    st.session_state['nombreModelo'] = None  # Nombre del modelo
    st.session_state['modelo'] = None  # Modelo de predicción
    st.session_state['keyUploader'] = 0  # Clave para el uploader de archivos y poder resetearlo
    st.session_state['listaImagenes'] = None # Lista con las imágenes cargadas
    st.session_state['listaImagenesProcesadas'] = None # Lista con las imágenes procesadas
    st.session_state['nuevaListaImagenes'] = False # Indica si se ha cargado una nueva lista de imágenes
    st.session_state['imagenesAProcesar'] = False  # Indica si se ha dado la orden de procesar las imágenes
    st.session_state['imagenesProcesadas'] = False  # Indica si las imagenes han sido procesadas
    
    st.session_state['umbral'] = None # Indica el umbral y permite detectar si se cambia
    st.session_state['overlap'] = None # Indica el solapamiento y permite detectar si se cambia
    st.session_state['margins'] = None # Indica los márgenes y permite detectar si se cambian
    

@st.cache_data(ttl=3600)  # 1 hora de persistencia
def cargaParametrosConfiguracion(ficheroConfiguracion):
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

@st.cache_data(ttl=3600)
def cargaModeloH5(nombreModelo, urlModelo):
    """
    Carga un modelo a partir de un nombre h5 y su url y devuelve el modelo cargado.

    Args:
        nombreModelo (str): El nombre del modelo.
        urlModelo (str): La URL del archivo del modelo.

    Returns:
        El modelo h5 cargado.

    """
    ficheroModelo = nombreModelo# .strip('.') #+ '.h5'  # Elimina los puntos del nombre del modelo y añade la extensión .h5
    es.cargaArchivoDrive(urlModelo, ficheroModelo)
    return pred.cargaModelo(ficheroModelo)


st.cache_data(ttl=3600)  # 1 hora de persistencia
def cargaColecciones(path):
    coleccionesPath = [f for f in path.iterdir()]
    nombresColeccionesYPath = {}
    for path in coleccionesPath:
        parametros = es.cargaParametrosConfiguracionYAML(path.joinpath(settings.DESCRIPCION_COLECCION))
        nombresColeccionesYPath[parametros['nombre']] = path
    return nombresColeccionesYPath

def actualizaClaveUploader():
    st.session_state['keyUploader'] += 1

@st.cache_data(ttl=3600)  # 1 hora de persistencia
def creaRectangulos(tamImOriginal, tamSubimagen, solapamiento, margenes):
    """
    Es un envoltorio de la función creaRectangulos de graficosImagenes.py para cachear los resultados.

    Creates a list of rectangles with indices based on the given parameters.

    Args:
        tamImOriginal (tuple): The size of the original image.
        tamSubimagen (tuple): The size of the subimage.
        solapamiento (int): The amount of overlap between subimages.
        margenes (int): The margin size around the subimages.

    Returns:
        list: A list of rectangles e indices. Los indices reflejan posicón del rectángulo en la imagen original
    """
    listaRectangulosConIndices = gi.creaListaRectangulosConIndices(tamImOriginal, tamSubimagen, solapamiento, margenes)
    return [rectangulo for rectangulo, _ in listaRectangulosConIndices]







def cargaImagen(): 
    """
    Loads an image and stores it in the session state.
    Save that the image has not been processed in the session state.
    Parameters:
        None

    Returns:
        None
    """
    st.session_state['imagen'] = pred.cargaImagen(st.session_state['imFile'])

    st.session_state['imagenSinProcesar'] = True






def hayModelo():
    return st.session_state['modelo'] is not None

def hayImagen():
    return st.session_state['imagen'] is not None

def hayImagenes():
    return st.session_state['hayImagenes'] is not None




def creaImagenConEtiquetas(predicciones, listaRectangulos, umbral, ficheroXML):
    imagenConHumanos = creaImagenConHumanos(predicciones, listaRectangulos, umbral)
    listaRectangulosEtiquetados = es.rectangulosEtiquetados(ficheroXML)
    imagenConHumanos = gi.dibujaRectangulos(imagenConHumanos, listaRectangulosEtiquetados, ancho=6, color='blue')
    return imagenConHumanos


def creaImagenConHumanos(predicciones, listaRectangulos, umbral):
    listaRectangulosConHumanos = []
    for i, prediccion in enumerate(predicciones):
        if prediccion + 0.005 >= umbral:  # 0.005 para redondear al alza y que el slider a 1.00 con predicción a 0.998, por ejemplo, de como encontrado humano
            listaRectangulosConHumanos.append(listaRectangulos[i])
    imagenConHumanos = gi.dibujaRectangulos(st.session_state['imagenGrid'], listaRectangulosConHumanos)
    return imagenConHumanos


def mapaCalor(predicciones, numImFil, numImCol):
    return gi.mapaCalor(predicciones, numImFil, numImCol)






def selectFolder():
    imagenesPath = ud.seleccionaDirectorio()
    if imagenesPath != None:
        listaImagenesPaths = ud.obtienePathFicheros(imagenesPath, st.session_state['imageExtensions'])
        if len(listaImagenesPaths) == 0:
            st.session_state['directorio'] =  -1
        else:
            st.session_state['directorio'] =  listaImagenesPaths
    else:
        st.session_state['directorio'] =  -2

def get_image_files():
        return [f for f in settings.PATH_COLECCIONESIMAGENES.iterdir() if f.suffix == '.JPG']



def cambiaEstadoProcesarImagenes():
    st.session_state['imagenesAProcesar'] = True

def main():
    
    if 'estadosInicializados' not in st.session_state:
        inicializaEstadosSesion()

    
    # Se hace str() al PATH porque st.logo() espera un string no un objeto Path
    st.logo(str(settings.PATH_LOGO), link=settings.URL_LOGO)
    st.subheader("Detección de Personas en Operaciones de Búsqueda y Rescate con UAS")

    
    parametros = cargaParametrosConfiguracion(settings.PATH_PARAMETROS)
    if parametros == None:
        st.error("Error cargando el fichero de configuración de parámetros.")
        st.stop()


    enlacesModelos = cargaEnlacesModelos()
    nombreModelo = wd.selectboxSeleccionModelo(enlacesModelos.keys())
    
    if nombreModelo is not None:
        if nombreModelo != st.session_state['nombreModelo']: # Se seleccionó un nuevo modelo
            st.session_state['nombreModelo'] = nombreModelo
            urlModelo = enlacesModelos[nombreModelo]
            st.session_state['modelo'] = cargaModeloH5(nombreModelo, urlModelo)
            st.session_state['imagenesAProcesar'] = False
            st.session_state['imagenesProcesadas'] = False         
    else:
        st.stop()  # Si no hay modelo elegido, se detiene la ejecución


    if st.session_state['overlap'] is None:
        st.session_state['overlap'] = parametros['imagenes']['overlap']
    if st.session_state['margins'] is None:
        st.session_state['margins'] = parametros['imagenes']['margins']
    if st.session_state['umbral'] is None:
        st.session_state['umbral'] = parametros['sliderUmbralPrediccion'][2]
    # La modificación de los ajustes implica hacer una nueva predicción
    umbral, overlap, margins = wd.seleccionAjustes(parametros)
    
    if overlap != st.session_state['overlap']:
        st.write('overlap')
        st.session_state['overlap'] = overlap
        st.session_state['imagenesProcesadas'] = False 
    if margins != st.session_state['margins']:
        st.write('margins', margins, st.session_state['margins'])
        st.session_state['margins'] = margins
        st.session_state['imagenesProcesadas'] = False 
    if umbral != st.session_state['umbral']:
        st.write('umbral')
        st.session_state['umbral'] = umbral
        st.session_state['imagenesProcesadas'] = False 







    nombresColeccionesYPath = cargaColecciones(settings.PATH_COLECCIONESIMAGENES)
    
    

    with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                pathImagenes = st.file_uploader("Selecciona una colección de imágenes archivo de imagen",
                                         accept_multiple_files=True,
                                         type=['jpg', 'jpeg', 'png'],
                                         key=f"uploader_{st.session_state['keyUploader']}")
                if len(pathImagenes):
                    st.session_state['listaImagenes'] = []
                    for pathImagen in pathImagenes:
                        st.session_state['listaImagenes'].append(es.cargaImagen(pathImagen))
                    actualizaClaveUploader()
                    st.session_state['nuevaListaImagenes'] = True
                    st.session_state['imagenesProcesadas'] = False
                    st.session_state['selectBoxColeccionImagenes'] = None
                    st.rerun() 
                    
                    
                    
            with col2:
                nombreColeccion = wd.selectboxColeccionImagenes(nombresColeccionesYPath)
                if nombreColeccion is not None:
                    pathColeccion = nombresColeccionesYPath[nombreColeccion]
                    pathImagenes = [str(f) for f in pathColeccion.iterdir() if f.suffix == '.JPG']
                    st.session_state['listaImagenes'] = []
                    for pathImagen in pathImagenes:
                        st.session_state['listaImagenes'].append(es.cargaImagen(pathImagen))
                    st.session_state['nuevaListaImagenes'] = True
                    st.session_state['imagenesProcesadas'] = False
                    nombreColeccion = None

    hayImagenesSinProcesar = st.session_state['nuevaListaImagenes'] and not st.session_state['imagenesProcesadas']
    hayImagenesProcesadas = st.session_state['nuevaListaImagenes'] and st.session_state['imagenesProcesadas']

    
        
    if hayImagenesSinProcesar:
        st.button("Procesar imágenes", key='buttonPredice', on_click=cambiaEstadoProcesarImagenes)
        n = st.number_input("Selecciona número de columnas de visualización", 1, 5, 3, key='numberInputNumColumnas1')
        wd.creaTablaImagenes(st.session_state['listaImagenes'], n)  

        if st.session_state['imagenesAProcesar']:
            st.session_state['imagenesAProcesar'] = False
            st.session_state['listaImagenesProcesadas'] = []
            numImagenes = len(st.session_state['listaImagenes'])
            for i, imagen in enumerate(st.session_state['listaImagenes']):
                listaRectangulos = creaRectangulos(imagen.size, parametros['imagenes']['size'], overlap, margins)
                with st.spinner(f'Procesando imagen {i+1} de {numImagenes}'):
                    st.session_state['predicciones'] = pred.predice(st.session_state['modelo'], imagen, listaRectangulos)
                    st.session_state['imagenGrid'] = gi.dibujaRejilla(imagen, listaRectangulos) 
                    st.session_state['listaImagenesProcesadas'].append(creaImagenConHumanos(st.session_state['predicciones'], listaRectangulos, umbral) )
            st.session_state['imagenesProcesadas'] = True
            st.rerun()
    
    if hayImagenesProcesadas:
        n = st.number_input("Selecciona número de columnas de visualización", 1, 5, 3, key='numberInputNumColumnas2')     
        wd.creaTablaImagenes(st.session_state['listaImagenesProcesadas'], n)     
        

        
           

    # st.stop()  
    # if listaImagenes1:
    #     wd.creaTablaImagenes(listaImagenes1)  
    

   
    
    

    

    
    
    
    # if hayModelo():
        
            
    #     if hayImagenes(): 
    #         wd.creaTablaImagenes()  
    #     if hayImagen():
    #         listaRectangulos, numImFil, numImCol = creaRectangulos(st.session_state['imagen'].size, parametros['imagenes']['size'], st.session_state['overlap'])
    #         if nuevaPrediccion or st.session_state['imagenSinProcesar']:
    #             with st.spinner('Procesando...'):
    #                 st.session_state['predicciones'] = pred.predice(st.session_state['modelo'], st.session_state['imagen'], listaRectangulos)
    #                 if st.session_state['imagenSinProcesar']:
    #                     st.session_state['imagenGrid'] = gi.dibujaRejilla(st.session_state['imagen'], listaRectangulos)

    #                 nuevaPrediccion = False
    #                 st.session_state['imagenSinProcesar'] = False
                
            
    #         if st.session_state['muestraEtiquetas']:
    #             ficheroXML = es.buscaFichero(st.session_state['directorio'], st.session_state['imFile'].name, 'xml')
    #             if ficheroXML is not None:
    #                 imagenProcesada = creaImagenConEtiquetas(st.session_state['predicciones'], listaRectangulos, umbral, ficheroXML)
    #                 st.image(imagenProcesada)
    #             else:
    #                 st.warning("No se ha encontrado el fichero de etiquetas.")
    #                 imagenProcesada = creaImagenConHumanos(st.session_state['predicciones'], listaRectangulos, umbral) 
    #                 st.image(imagenProcesada)
    #         else:
    #             imagenProcesada = creaImagenConHumanos(st.session_state['predicciones'], listaRectangulos, umbral) 
    #             st.image(imagenProcesada)
            
    #         figura = mapaCalor(st.session_state['predicciones'], numImFil, numImCol)
                
    #         st.pyplot(figura)
            

    
if __name__ == '__main__':
    main()  