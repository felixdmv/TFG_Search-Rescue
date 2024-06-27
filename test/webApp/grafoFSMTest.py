import io
from PIL import Image as PILImage
import streamlit as st
from transitions.extensions import GraphMachine
from transitions import Machine

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

def show_graph(machine, **kwargs):
    """Muestra el gráfico de la máquina de estados en Streamlit."""
    stream = io.BytesIO()
    graph = machine.get_graph(**kwargs)
    graph.draw(stream, prog='dot', format='png')
    stream.seek(0)
    img = PILImage.open(stream)
    st.image(img)

# Crear el objeto de la máquina de estados
app_state_machine = AppStateMachine()

# Configurar la máquina de estados para que use GraphMachine
app_state_machine.machine = GraphMachine(model=app_state_machine, states=AppStateMachine.states, initial='inicio')
# app_state_machine.machine.add_transition('reset', '*', 'inicio')
# app_state_machine.machine.add_transition('excepcion', '*', 'excepcion')
app_state_machine.machine.add_transition('cargaModelo', ['inicio', 'hayModelo', 'hayImagenes'], 'cargandoModelo')
app_state_machine.machine.add_transition('cargandoModelo_hayModelo', 'cargandoModelo', 'hayModelo')
app_state_machine.machine.add_transition('cargandoModelo_hayImagenes', 'cargandoModelo', 'hayImagenes')
app_state_machine.machine.add_transition('hayModelo_hayImagenes', 'hayModelo', 'hayImagenes')
app_state_machine.machine.add_transition('hayImagenes_predicciones', 'hayImagenes', 'predicciones')
app_state_machine.machine.add_transition('predicciones_imagenesProcesadas', 'predicciones', 'imagenesProcesadas')
app_state_machine.machine.add_transition('imagenesProcesadas_hayImagenes', 'imagenesProcesadas', 'hayImagenes')

# Iniciar la aplicación de Streamlit
st.title("Diagrama de Estados de la Aplicación")

if st.button('Mostrar Diagrama de Estados'):
    show_graph(app_state_machine.machine)
