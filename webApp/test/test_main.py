from settings_test import PATH_MAIN
from streamlit.testing.v1 import AppTest
import pytest


def subHeader(at):
    assert len(at.subheader) == 1    
    assert "Detección de Personas en Operaciones de Búsqueda y Rescate con UAS" == at.subheader[0].value

def selectboxModelo(at):
    assert at.selectbox('selectboxModelo').value == None
    at.selectbox('selectboxModelo').select_index(0)
    assert at.selectbox('selectboxModelo').value == 'modelo1'
    at.selectbox('selectboxModelo').select_index(1)
    assert at.selectbox('selectboxModelo').value == 'modelo2'

def expanderSeleccionAjustes(at):# Verificamos que hay un expander con 5 sliders y 1 botón
    assert len(at.expander) == 1
    assert len(at.expander[0].slider) == 5
    assert len(at.expander[0].button) == 1
    assert at.expander[0].label == 'Selección de ajustes'
    
    # No estoy seguro de que índice tiene cada slider. Lo identifico por su label
    # Compruebo que el valor inicial es el que se ha cargado en el estado de la sesión
    # Se modifica el valor de cada slider y se comprueba que se ha modificado en el estado de la sesión
    # tras pulsar el botón del formulario
    sliders = at.expander[0].slider
    for slider in sliders:
        if slider.label == 'Solapamiento horizontal':
            assert slider.value == at.session_state['overlap'][0]
            slider.set_value(slider.value+1)
        elif slider.label == 'Solapamiento vertical':
            assert slider.value == at.session_state['overlap'][1]
            slider.set_value(slider.value+1)
        elif slider.label == 'Margen horizontal':
            assert slider.value == at.session_state['margins'][0]
            slider.set_value(slider.value+1)
        elif slider.label == 'Margen vertical':
            assert slider.value == at.session_state['margins'][1]
            slider.set_value(slider.value+1)
        elif slider.label == 'Umbral de predicción':
            assert slider.value == at.session_state['sliderUmbralPrediccion'][2]
            slider.set_value(slider.value+0.1)
        else:
            assert False, f'El slider {slider.label} no debería estar aquí'

    # Simulamos un click() en el botón del formulario
    at.expander[0].button[0].click()
    for slider in sliders:
        if slider.label == 'Solapamiento horizontal':
            assert slider.value == at.session_state['overlap'][0] + 1
        elif slider.label == 'Solapamiento vertical':
            assert slider.value == at.session_state['overlap'][1] + 1
        elif slider.label == 'Margen horizontal':
            assert slider.value == at.session_state['margins'][0] + 1
        elif slider.label == 'Margen vertical':
            assert slider.value == at.session_state['margins'][1] + 1
        elif slider.label == 'Umbral de predicción':
            assert slider.value == at.session_state['sliderUmbralPrediccion'][2] + 0.1
        else:
            assert False, f'El slider {slider.label} no debería estar aquí'

@pytest.fixture()
def at():
    """Fixture that prepares the Streamlit app tests"""
    at = AppTest.from_file(str(PATH_MAIN), default_timeout=30)
    at.secrets["modelos"] =[["modelo1", "a"],
                ["modelo2", "b"]]
    yield at.run()


def test_estadoInicial(at):
    subHeader(at)
    selectboxModelo(at)

def test_estadoHayModelo(at):
    at.session_state.fsm.state = 'hayModelo'
    at.run()
    subHeader(at)
    selectboxModelo(at)
    expanderSeleccionAjustes(at)   
    print('----->', len(at.selectbox))
        
def test_estadoHayImagenes(at):
    at.session_state.fsm.state = 'hayImagenes'
    at.run()

def test_estadoHayPrediccion(at):
    pass


    
    

if __name__ == '__main__':
    pytest.main(['-s'])