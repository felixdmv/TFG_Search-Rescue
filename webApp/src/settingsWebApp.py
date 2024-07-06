from pathlib import Path
import sys

# Obtiene la ruta absoluta del fichero actual
FILE = Path(__file__).resolve()
# Obtiene el directorio dos niveles por encima del directorio actual
ROOT_TFG = FILE.parents[2]
ROOT_WEBAPP = FILE.parents[1]

# Añade la ruta raíz a la lista sys.path si no está ya ahí
if ROOT_TFG not in sys.path:
    sys.path.append(str(ROOT_TFG))
if ROOT_WEBAPP not in sys.path:
    sys.path.append(str(ROOT_WEBAPP))


PATH_PARAMETROS = ROOT_WEBAPP.joinpath('config', 'parametros.yaml')
PATH_COLECCIONESIMAGENES = ROOT_WEBAPP.joinpath('coleccionesImagenes')
PATH_LOGO = ROOT_WEBAPP.joinpath('static', 'logoGICAP.jpg')

URL_LOGO = 'https://gicap.ubu.es/main/home.shtml'

DESCRIPCION_COLECCION = 'descripcion.yaml'