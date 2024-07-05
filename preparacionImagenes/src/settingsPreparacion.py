from pathlib import Path
import sys

# Obtiene la ruta absoluta del fichero actual
FILE = Path(__file__).resolve()
# Obtiene el directorio dos niveles por encima del directorio actual
ROOT_TFG = FILE.parents[2]
ROOT_PREPARACIONIMAGENES = FILE.parents[1]
# Añade la ruta raíz a la lista sys.path si no está ya ahí
if ROOT_TFG not in sys.path:
    sys.path.append(str(ROOT_TFG))
if ROOT_PREPARACIONIMAGENES not in sys.path:
    sys.path.append(str(ROOT_PREPARACIONIMAGENES))

PATH_PARAMETROS = ROOT_PREPARACIONIMAGENES.joinpath('config', 'parametros.yaml')
PATH_INFORMEANALISISFICHEROS = ROOT_PREPARACIONIMAGENES.joinpath('informes', 'informeFicheros.txt')
PATH_INFORMEBND = ROOT_PREPARACIONIMAGENES.joinpath('informes', 'informeBnd.txt')
PATH_INFORMECREACIONSUBIMAGENES = ROOT_PREPARACIONIMAGENES.joinpath('informes', 'informeCreacionSubimagenes.txt')