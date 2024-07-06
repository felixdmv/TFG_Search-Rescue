from settings_test import PATH_MAIN
from streamlit.testing.v1 import AppTest
import pytest


def subHeader(at):
    """
    This function verifies the subheader value in the given object.

    Parameters:
    at (object): The object containing the subheader value.

    Returns:
    None
    """
    assert len(at.subheader) == 1    
    assert "Detección de Personas en Operaciones de Búsqueda y Rescate con UAS" == at.subheader[0].value


def selectboxModelo(at):
    """
    This function tests the behavior of a selectbox widget.

    Parameters:
    - at: The instance of the application under test.

    Returns:
    None
    """
    assert at.selectbox('selectboxModelo').value == None
    at.selectbox('selectboxModelo').select_index(0)
    assert at.selectbox('selectboxModelo').value == 'modelo1'
    at.selectbox('selectboxModelo').select_index(1)
    assert at.selectbox('selectboxModelo').value == 'modelo2'


def expanderSeleccionAjustes(at):
    """
    Verifies that there is an expander with 5 sliders and 1 button.
    Checks the initial values of the sliders and updates them.
    Clicks the button and verifies the updated values of the sliders.

    Args:
        at: An object representing the application under test.

    Raises:
        AssertionError: If the number of expanders, sliders, or buttons is incorrect,
                        or if a slider label is unexpected.

    Returns:
        None
    """
    assert len(at.expander) == 1
    assert len(at.expander[0].slider) == 5
    assert len(at.expander[0].button) == 1
    assert at.expander[0].label == 'Selección de ajustes'
    
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
    """
    This function tests the initial state of the application.

    Parameters:
    - at: The test object.

    Returns:
    None
    """
    subHeader(at)
    selectboxModelo(at)


def test_estadoHayModelo(at):
    """
    Test function for the 'estadoHayModelo' state.

    Args:
        at: The test object.

    Returns:
        None
    """
    at.session_state.fsm.state = 'hayModelo'
    at.run()
    subHeader(at)
    selectboxModelo(at)
    expanderSeleccionAjustes(at)   
    print('----->', len(at.selectbox))
   
        
def test_estadoHayImagenes(at):
    """
    Test case for the 'estadoHayImagenes' function.

    This test sets the state of the 'fsm' object in the 'at.session_state' to 'hayImagenes',
    and then runs the 'at' object.

    Args:
        at: The object representing the test case.

    Returns:
        None
    """
    at.session_state.fsm.state = 'hayImagenes'
    at.run()


def test_estadoHayPrediccion(at):
    """
    This function tests the estadoHayPrediccion method.

    Parameters:
    - at: The input parameter for the estadoHayPrediccion method.

    Returns:
    - None
    """
    pass


if __name__ == '__main__':
    pytest.main(['-s'])