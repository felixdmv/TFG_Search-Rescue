import utils.entradaSalida as es
import prediccion as pred
import streamlit as st

#######################################################################################################################
def ocultaNombreFichero():
    """
    Hides the file name in the file uploader component.

    This function adds custom CSS to hide the file name in the file uploader component.
    It sets the display property to 'none' and visibility property to 'hidden' for the file name element.

    Parameters:
        None

    Returns:
        None
    """
    css = """
            .stFileUploaderFile {
                display: none;
                visibility: hidden;
            }
            """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
#######################################################################################################################

#######################################################################################################################
def flipMuestraEtiquetas():
    st.session_state['muestraEtiquetas'] = not st.session_state['muestraEtiquetas']

def selectFolder():
    st.session_state['directorio'] = es.seleccionaDirectorio()

def muestraEtiquetas():
    col1, col2 = st.columns(2)
        
    with col1:
        if st.session_state['directorio'] is None:
            st.checkbox("Muestra imagen con etiquetas", help='Debes seleccionar previamente un directorio de imágenes', value=False, disabled=True)
            st.session_state['muestraEtiquetas'] = False
        else:
            st.checkbox("Muestra imagen con etiquetas", help=f"Se buscarán los archivos de etiquetas desde {st.session_state['directorio']}", value=False, on_change=flipMuestraEtiquetas)
    with col2:
        st.button("Selecciona directorio de etiquetas", on_click=selectFolder)
        if st.session_state['directorio'] is not None:
            st.write("Directorio:", st.session_state['directorio'])
        else:
            st.write("Directorio no seleccionado")
#######################################################################################################################

#######################################################################################################################
def sliderSolapamientos(size, solapamiento):
    width, height = size
    solap_x, solap_y = solapamiento
    
    col1, col2 = st.columns(2)
    with col1:
        solape_x = st.slider('Solapamiento horizontal', 0, width, solap_x, 1)
    with col2:
        solape_y = st.slider('Solapamiento vertical', 0, height, solap_y, 1)

    return solape_x, solape_y
#######################################################################################################################

#######################################################################################################################
def sliderMargenes(solapamiento, margenes):
    margen_x, margen_y = margenes
    solap_x, solap_y = solapamiento
    
    col1, col2 = st.columns(2)
    with col1:
        margen_x = st.slider('Margen horizontal', 0, solap_x, margen_x, 1)
    with col2:
        margen_y = st.slider('Margen vertical', 0, solap_y, margen_y, 1)

    return margen_x, margen_y
#######################################################################################################################

#######################################################################################################################
def cargaColeccion():
    pass # Sin implementar

def selectboxColeccionImagenes(listaColecciones):
    with st.expander("Descarga de colección de imágenes", expanded=False):
        st.selectbox(label='Elige una colección', key='ficheroImagenes', index=None, options=listaColecciones, on_change=cargaColeccion)
#######################################################################################################################

#######################################################################################################################
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
    ficheroModelo = nombreModelo.strip('.') + '.h5'  # Elimina los puntos del nombre del modelo y añade la extensión .h5
    es.cargaArchivoDrive(urlModelo, ficheroModelo)
    return pred.cargaModelo(ficheroModelo)


def selectboxSeleccionModelo(enlacesModelos):
    nombreModelo = st.selectbox(label='Selecciona un modelo', key='selectboxModelo', index=None, options=enlacesModelos.keys())
    if nombreModelo is not None:
        if nombreModelo != st.session_state['nombreModelo']: # Se seleccionó un nuevo modelo
            st.session_state['nombreModelo'] = nombreModelo
            st.session_state['nuevoModelo'] = True
            urlModelo = enlacesModelos[nombreModelo]
            st.session_state['modelo'] = cargaModeloH5(nombreModelo, urlModelo)
#######################################################################################################################

#######################################################################################################################
def seleccionAjustes(parametros):
    with st.expander("Selección de ajustes", expanded=True):
        with st.container(border=True):
            # *parametros['sliderUmbralPrediccion'] unpacks the tuple into individual arguments
            umbral = st.slider('Umbral de predicción', *parametros['sliderUmbralPrediccion'])
        with st.container(border=True):
            solape_x, solape_y = sliderSolapamientos(parametros['imagenes']['size'], parametros['imagenes']['overlap'])
            margen_x, margen_y = sliderMargenes(parametros['imagenes']['overlap'], parametros['imagenes']['margins'])
        with st.container(border=True):
            muestraEtiquetas()
    return umbral, solape_x, solape_y, margen_x, margen_y
#######################################################################################################################