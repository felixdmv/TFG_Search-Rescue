import settings
import utils.entradaSalida as es
import utils.graficosImagenes as gi
import utils.utilidadesDirectorios as ud
import prediccion as pred
import widgets as wd
import utils.entradaSalida as es
import streamlit as st
import imageGrid as ig

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



def inicializaEstadosSesion():
    """
    Initializes the session states.

    This function sets the initial values for various session states used in the application.
    """
    st.session_state['estadosInicializados'] = True # Indica si los estados de la sesión han sido inicializados
    st.session_state['directorio'] = None  # Elección de directorio de trabajo de imágenes y etiquetas
    st.session_state['nombreModelo'] = None  # Nombre del modelo
    st.session_state['modelo'] = None  # Modelo de predicción
    st.session_state['nuevoModelo'] = False  # Indica si se ha cargado un nuevo modelo
    st.session_state['imagen'] = None  # Imagen a procesar
    st.session_state['imagenSinProcesar'] = False  # Indica si la imagen ha sido procesada
    st.session_state['imagenGrid'] = None  # Imagen con la rejilla de rectángulos
    st.session_state['muestraEtiquetas'] = False  # Indica si se muestran las etiquetas
    st.session_state['overlap'] = None # Indica el solapamiento y permite detectar si se cambia
    st.session_state['margins'] = None # Indica los márgenes y permite detectar si se cambian
    st.session_state['imageExtensions'] = None  # Extensiones de imagen permitidas



def hayModelo():
    return st.session_state['modelo'] is not None

def hayImagen():
    return st.session_state['imagen'] is not None


@st.cache_data(ttl=3600)  # 1 hora de persistencia
def creaRectangulos(tamImOriginal, tamSubimagen, solapamiento):
    """
    Es un envoltorio de la función creaRectangulos de graficosImagenes.py para cachear los resultados.
    Creates a list of rectangles based on the original image size, subimage size, and overlap.
    Args:
        tamImOriginal (tuple): The size of the original image in the format (width, height).
        tamSubimagen (tuple): The size of the subimage in the format (width, height).
        solapamiento (float): The overlap between subimages

    Returns:
        tuple: A tuple containing the list of rectangles, the number of subimages in the vertical direction, and the number of subimages in the horizontal direction.
    """
    return gi.creaRectangulos(tamImOriginal, tamSubimagen, solapamiento)


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


@st.cache_data(ttl=3600)  # 1 hora de persistencia
def cargaParametrosConfiguracion(ficheroConfiguracion):
    return es.cargaParametrosConfiguracionYAML(ficheroConfiguracion)

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

def main():
    # wd.ocultaNombreFichero()
    if 'estadosInicializados' not in st.session_state:
        inicializaEstadosSesion()

    st.logo('webApp/static/logoGICAP.png', link='https://gicap.ubu.es/main/grupo.shtml')
    st.title("Detección de Personas en Operaciones de Búsqueda y Rescate con UAS")

    ig.demoGrid()

    # output = 'AlexNet.5.modelo.h5'
    # urlModelo = "https://drive.google.com/uc?id=15PgOiHfvLpxlyg5KtMzwXW-SsoNNI0Ok"
    # gdown.download(urlModelo, output, quiet=False)
    # st.session_state['modelo'] = pred.cargaModelo1(output)
    # st.session_state['nuevoModelo'] = True
    
    parametros = cargaParametrosConfiguracion('webapp/config/parametros.yaml')
    if parametros == None:
        st.error("Error cargando el fichero de configuración de parámetros.")
        st.stop()


    enlacesModelos = cargaEnlacesModelos()

    
    if st.session_state['overlap'] is None:
        st.session_state['overlap'] = parametros['imagenes']['overlap']
    if st.session_state['margins'] is None:
        st.session_state['margins'] = parametros['imagenes']['margins']
    if st.session_state['imageExtensions'] is None:
        st.session_state['imageExtensions'] = parametros['imagenes']['imageExtensions']


    wd.selectboxSeleccionModelo(enlacesModelos)

    

    # La modificación de los ajustes implica hacer una nueva predicción
    umbral, solape_x, solape_y, margen_x, margen_y = wd.seleccionAjustes(parametros)
    if [solape_x, solape_y] != st.session_state['overlap']:
        st.session_state['nuevoModelo'] = True
        st.session_state['imagenSinProcesar'] = True
        st.session_state['overlap'] = [solape_x, solape_y]
    if [margen_x, margen_y] != st.session_state['margins']:
        st.session_state['nuevoModelo'] = True
        st.session_state['imagenSinProcesar'] = True
        st.session_state['margins'] = [margen_x, margen_y]
    
    
    if hayModelo():
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.button("Selecciona directorio de imágenes", key="1", on_click=selectFolder)
                if st.session_state['directorio']  == -1:
                    st.warning("Directorio sin imágenes válidas")
                elif st.session_state['directorio']  == -2:
                    st.warning("No se ha seleccionado ningún directorio")
                else:
                    pass
                
                    
            with col2:
                wd.selectboxColeccionImagenes(parametros['coleccionesImagenes'])
            
            
            
        if hayImagen():
            listaRectangulos, numImFil, numImCol = creaRectangulos(st.session_state['imagen'].size, parametros['imagenes']['size'], st.session_state['overlap'])
            if st.session_state['nuevoModelo'] or st.session_state['imagenSinProcesar']:
                with st.spinner('Procesando...'):
                    st.session_state['predicciones'] = pred.predice(st.session_state['modelo'], st.session_state['imagen'], listaRectangulos)
                    if st.session_state['imagenSinProcesar']:
                        st.session_state['imagenGrid'] = gi.dibujaRejilla(st.session_state['imagen'], listaRectangulos)

                    st.session_state['nuevoModelo'] = False
                    st.session_state['imagenSinProcesar'] = False
                
            
            if st.session_state['muestraEtiquetas']:
                ficheroXML = es.buscaFichero(st.session_state['directorio'], st.session_state['imFile'].name, 'xml')
                if ficheroXML is not None:
                    imagenProcesada = creaImagenConEtiquetas(st.session_state['predicciones'], listaRectangulos, umbral, ficheroXML)
                    st.image(imagenProcesada)
                else:
                    st.warning("No se ha encontrado el fichero de etiquetas.")
                    imagenProcesada = creaImagenConHumanos(st.session_state['predicciones'], listaRectangulos, umbral) 
                    st.image(imagenProcesada)
            else:
                imagenProcesada = creaImagenConHumanos(st.session_state['predicciones'], listaRectangulos, umbral) 
                st.image(imagenProcesada)
            
            figura = mapaCalor(st.session_state['predicciones'], numImFil, numImCol)
                
            st.pyplot(figura)
            

    
if __name__ == '__main__':
    main()  