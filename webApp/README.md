# webApp
- Crear carpeta del proyecto --> tfg
- Inicilaizar git dentro de la carpeta --> `git init`
- Crear un entorno virtual desde Anaconda Prompt --> `conda create -n tfg`
- Activar el entorno virtual --> `conda activate tfg`
- Instalar `streamlit` en el entorno virtual --> `pip install streamlit`
- Verificar que la instalación es OK --> `streamlit hello`
- Instalar `tensorflow` --> `pip install tensorflow`
- Instalar `matplotlib` para mapa de calor --> `pip install matplotlib`
- Instalar `seaborn` para mapa de calor --> `pip install seaborn`
- Instalar `gdown`para cargar ficheros desde Google Drive --> 

## VISUAL STUDIO CODE
- Select `Interpreter command` from the _Command Palette (Ctrl+Shift+P)_ --> Python (tfg) instalará probablemente última versión estable de Python
- Crear `.gitignore` con las exclusiones habituales para no seguimiento de _git_. Se añade `*.h5` debido a su elevado tamaño

## EJECUCIÓN EN LÍNEA DE COMANDOS
`streamlit run main.py`
