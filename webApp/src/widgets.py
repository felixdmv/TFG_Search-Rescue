import utils.entradaSalida as es
import streamlit as st


#######################################################################################################################
def selectboxSeleccionModelo(nombresModelos):
    return st.selectbox(label='Selecciona un modelo', key='selectboxModelo', index=None, options=nombresModelos)
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
def seleccionAjustes(parametros):
    with st.expander("Selección de ajustes", expanded=True):
        with st.container(border=True):
            # *parametros['sliderUmbralPrediccion'] unpacks the tuple into individual arguments
            umbral = st.slider('Umbral de predicción', *parametros['sliderUmbralPrediccion'])
        with st.container(border=True):
            solapamiento = sliderSolapamientos(parametros['imagenes']['size'], parametros['imagenes']['overlap'])
            margenes = sliderMargenes(parametros['imagenes']['overlap'], parametros['imagenes']['margins'])
    return umbral, solapamiento, margenes
#######################################################################################################################






#######################################################################################################################
# def flipMuestraEtiquetas():
#     st.session_state['muestraEtiquetas'] = not st.session_state['muestraEtiquetas']

# def selectFolder():
#     st.session_state['directorio'] = es.seleccionaDirectorio()

# def muestraEtiquetas():
#     col1, col2 = st.columns(2)
        
#     with col1:
#         if st.session_state['directorio'] is None:
#             st.checkbox("Muestra imagen con etiquetas", help='Debes seleccionar previamente un directorio de imágenes', value=False, disabled=True)
#             st.session_state['muestraEtiquetas'] = False
#         else:
#             st.checkbox("Muestra imagen con etiquetas", help=f"Se buscarán los archivos de etiquetas desde {st.session_state['directorio']}", value=False, on_change=flipMuestraEtiquetas)
#     with col2:
#         st.button("Selecciona directorio de etiquetas", on_click=selectFolder)
#         if st.session_state['directorio'] is not None:
#             st.write("Directorio:", st.session_state['directorio'])
#         else:
#             st.write("Directorio no seleccionado")
#######################################################################################################################



#######################################################################################################################
def cargaColeccion(nombresColeccionesYPath):
    pathColeccion = nombresColeccionesYPath[st.session_state['selectBoxColeccionImagenes']]
    st.session_state['hayImagenes'] = True
    st.session_state['listaImagenes'] = [str(f) for f in pathColeccion.iterdir() if f.suffix == '.JPG']


def selectboxColeccionImagenes(nombresColeccionesYPath):
    listaNombres = list(nombresColeccionesYPath.keys())
    return st.selectbox(label='Elige una colección', key='selectBoxColeccionImagenes', index=None, options=listaNombres)
        
#######################################################################################################################


#######################################################################################################################
def creaTablaImagenes(listaImagenes, n):
    grupos = []
    for i in range(0, len(listaImagenes), n):
        grupos.append(listaImagenes[i:i+n])

    for grupo in grupos:
        cols = st.columns(n)
        for i, imagen in enumerate(grupo):
            cols[i].image(imagen)
#######################################################################################################################
