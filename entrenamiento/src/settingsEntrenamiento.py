from pathlib import Path
import sys

# Obtiene la ruta absoluta del fichero actual
FILE = Path(__file__).resolve()
# Obtiene el directorio dos niveles por encima del directorio actual
ROOT_TRAINING = FILE.parents[1]
ROOT_SRC = ROOT_TRAINING.joinpath('src')

# Añade la ruta raíz a la lista sys.path si no está ya ahí
if ROOT_TRAINING not in sys.path:
    sys.path.append(str(ROOT_TRAINING))
if ROOT_SRC not in sys.path:
    sys.path.append(str(ROOT_SRC))

PATH_PARAMETROS = ROOT_SRC.joinpath('parametros.yaml')
PATH_MAIN = Path(__file__).parents[1].joinpath('src/main.py')