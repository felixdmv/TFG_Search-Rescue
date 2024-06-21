from pathlib import Path
import sys

# Obtiene la ruta absoluta del fichero actual
FILE = Path(__file__).resolve()
# Obtiene el directorio padre del directorio actual (que sería `modulo2`)
ROOT = FILE.parents[2]
# Añade la ruta raíz a la lista sys.path si no está ya ahí
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
