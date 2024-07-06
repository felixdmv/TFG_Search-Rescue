import utils.entradaSalida as es
import streamlit as st
import utils.utilidadesDirectorios as ud

# Se añaden los asteriscos por regla de estilo de Streamlit para separar los widgets entre sí

#######################################################################################################################
def selectboxSeleccionModelo(nombresModelos):
    """
    Returns a selectbox widget for selecting a model.

    Args:
        nombresModelos (list): A list of model names.

    Returns:
        str: The selected model name.

    """
    if st.session_state['nombreModelo'] is None:
        indice = None
    else:
        indice = nombresModelos.index(st.session_state['nombreModelo'])
    return st.selectbox(label='Selecciona un modelo', key='selectboxModelo', index=indice, options=nombresModelos)
#######################################################################################################################

#######################################################################################################################
def sliderSolapamientos(size, solapamiento):
    """
    Function to create sliders for adjusting the overlap values.

    Parameters:
    - size (tuple): A tuple containing the width and height of the area.
    - solapamiento (tuple): A tuple containing the initial overlap values.

    Returns:
    - tuple: A tuple containing the adjusted overlap values.

    Example usage:
    size = (800, 600)
    solapamiento = (100, 50)
    sliderSolapamientos(size, solapamiento)
    """
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
    """
    Function to create sliders for adjusting margins.

    Parameters:
    solapamiento (tuple): A tuple containing the maximum values for solap_x and solap_y.
    margenes (tuple): A tuple containing the current values for margen_x and margen_y.

    Returns:
    tuple: A tuple containing the updated values for margen_x and margen_y.
    """
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
def creaTablaImagenes(listaImagenes, n):
    """
    Crea una tabla de imágenes en la interfaz gráfica.

    Parameters:
    - listaImagenes (list): Una lista de imágenes.
    - n (int): El número de columnas en la tabla.

    Returns:
    None
    """
    grupos = []
    for i in range(0, len(listaImagenes), n):
        grupos.append(listaImagenes[i:i+n])

    for grupo in grupos:
        cols = st.columns(n)
        for i, imagen in enumerate(grupo):
            cols[i].image(imagen)
#######################################################################################################################

#######################################################################################################################
def cargaImagenesYNombres(pathsImagenes, local=True):
    """
    Carga las imágenes y los nombres de archivo en el estado de sesión.

    Args:
        pathsImagenes (list): Una lista de rutas de archivo de imágenes.
        local (bool, optional): Indica si se deben utilizar los nombres de archivo locales o los nombres de archivo base. 
                                Por defecto es True.

    Returns:
        None
    """
    st.session_state['listaImagenes'] = [es.cargaImagen(pathImagen) for pathImagen in pathsImagenes]
    if local:
        st.session_state['nombresImagenes'] = [pathImagen.name for pathImagen in pathsImagenes]
    else:
        st.session_state['nombresImagenes'] = [ud.obtieneNombreBase(pathImagen) for pathImagen in pathsImagenes]
    
    
def actualizaClaveFileUploader():
    '''
    Actualiza el valor de la clave del file_uploader para forzar un nuevo file_uploader.
    El problema se debe a que file_uploader retiene la lista de archivos seleccionados y no la actualiza
    sino que lo acumula.
    '''
    st.session_state['keyUploader'] += 1
    
    
def cargaImagenesLocal():
    """
    Function to load local images.

    Returns:
        None
    """
    pathImagenes= st.file_uploader("Selecciona imágenes",
                                            accept_multiple_files=True,
                                            type=['jpg', 'jpeg', 'png'],
                                            key=f"fileuploader_{st.session_state['keyUploader']}")
    if len(pathImagenes):
        cargaImagenesYNombres(pathImagenes)
        st.session_state.listaImagenesLocal = True
        st.session_state.imagenes = None  # Para que no haya dudas del origen de las imagenes
        actualizaClaveFileUploader()
        st.session_state.nombreColeccion = None  # Para que no haya dudas del origen de las imagenes
        st.session_state.selectBoxColeccionImagenes = None  # Para que se muestre el selectbox con la nueva colección
        st.rerun()  # Para forzar a mostrar un nuevo file_uploader
       
                
def actualizaColeccion(nombresColecciones):
    """
    Actualiza la colección de imágenes seleccionada en la aplicación.

    Args:
        nombresColecciones (list): Una lista de nombres de colecciones disponibles.

    Returns:
        None
    """
    nombreColeccion = st.session_state.selectBoxColeccionImagenes
    if nombreColeccion is not None:
        st.session_state.indiceColeccion = nombresColecciones.index(nombreColeccion)
        if st.session_state.nombreColeccion != nombreColeccion:
            st.session_state.nombreColeccion = nombreColeccion
            pathColeccion = st.session_state.nombresColeccionesYPath[nombreColeccion]
            pathImagenes = [str(f) for f in pathColeccion.iterdir() if f.suffix.lower() == '.jpg']
            cargaImagenesYNombres(pathImagenes, local=False)
            st.session_state.listaImagenesLocal = False
            
    else:
        st.session_state.indiceColeccion = None
        st.session_state.nombreColeccion = None


def cargaImagenesServidor():
    """
    Carga las imágenes del servidor y actualiza las variables de estado correspondientes.

    Returns:
        None
    """
    nombresColecciones = sorted(list(st.session_state.nombresColeccionesYPath.keys()))
    if st.session_state.nombreColeccion is None:
        st.session_state.indiceColeccion = None
    
    nombreColeccion = st.selectbox(label='Elige una colección',
                                                    key='selectBoxColeccionImagenes',
                                                    index=st.session_state.indiceColeccion,
                                                    options=nombresColecciones,
                                                    on_change=actualizaColeccion,
                                                    args=(nombresColecciones,))
    

    if st.session_state.listaImagenesLocal == False and st.session_state.nombreColeccion is None:
        st.session_state.listaImagenes = []
        st.session_state.nombresImagenes = []
        st.session_state.nombreColeccion = None
#######################################################################################################################

#######################################################################################################################
def formularioParametros():
    """
    Displays a form for selecting parameters and updates the session state with the selected values.

    Returns:
        None
    """
    with st.expander("Selección de ajustes"):
        with st.form("Selección de ajustes", border=False):
            with st.container(border=True):
                umbral = st.slider('Umbral de predicción', *st.session_state.sliderUmbralPrediccion)
            with st.container(border=True):
                solapamiento = sliderSolapamientos(st.session_state.sizeSubimagenes, st.session_state.overlap)
                margenes = sliderMargenes(st.session_state.overlap, st.session_state.margins)

            if st.form_submit_button("Actualizar parámetros"):
                st.session_state.sliderUmbralPrediccion[2] = umbral
                st.session_state.overlap = solapamiento
                st.session_state.margins = margenes
#######################################################################################################################