from pathlib import Path
import sys

# Obtiene la ruta absoluta del fichero actual
FILE = Path(__file__).resolve()
# Obtiene el directorio dos niveles por encima del directorio actual

ROOT_WEBAPP = FILE.parents[1]

ROOT_SRC = ROOT_WEBAPP.joinpath('src')


if ROOT_WEBAPP not in sys.path:
    sys.path.append(str(ROOT_WEBAPP))
if ROOT_SRC not in sys.path:
    sys.path.append(str(ROOT_SRC))

PATH_MAIN = Path(__file__).parents[1].joinpath('src/main.py')
