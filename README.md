# Search & Rescue - TFG

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado:  
  
"Sistema de visión artificial para detección de personas en entornos naturales a partir de imágenes tomadas desde un dron"  
  
Está organizado en tres subproyectos: preparación de imágenes, entrenamiento de redes neuronales convolucionales (CNNs) y una aplicación web (WebApp).
Se incluyen además scripts en utils que se usan desde varios subproyectos.
Cada subproyecto está cuidadosamente estructurado para facilitar su comprensión y reproducibilidad.


## Tecnologías Utilizadas

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YAML](https://img.shields.io/badge/yaml-000000?style=for-the-badge&logo=yaml&logoColor=white)](https://yaml.org/)
[![GitHub](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

## Estructura de Directorios

La estructura general del proyecto es la siguiente:

### Directorio Raíz

En el directorio raíz se encuentran las siguientes carpetas y archivos:

- **\_\_init\_\_.py**: Añade el directorio `utils` al path para permitir la importación de módulos.
- **pytest.ini**: Configura el framework de pruebas pytest, para la ejecución automática de pruebas.
- **.gitignore**: Especifica los archivos que Git debe ignorar.

### Subproyecto de Preparación de Imágenes

Este subproyecto se divide en:

- **config**: Contiene un archivo de configuración `.yaml` con parámetros del recorte, rutas y extensiones.
- **informes**: Almacena los análisis estadísticos del dataset antes y después del procesado.
- **src**: Scripts de recorte, reetiquetado y creación de CSVs para la validación cruzada. Incluye un archivo `settings` para configurar y añadir rutas al `sys.path`.

### Subproyecto de Entrenamiento de CNNs

Este subproyecto se organiza en:

- **codigo**: Script de entrenamiento de CNNs y archivo de configuración `.yaml`.
- **imagenes**: Subimágenes recortadas y CSVs para la validación cruzada.

Tras ejecutar el script de entrenamiento, se crea automáticamente una carpeta adicional:

- **resultados**: Almacena los resultados de los entrenamientos, métricas en cada época, métricas por umbrales, modelo entrenado `.h5` y parametrización elegida en `.yaml`.

Durante la ejecución, se crean y borran automáticamente dos carpetas:

- **datos\_entreno**: Separa las imágenes en entrenamiento, validación y prueba.
- **tmp**: Guarda archivos temporales durante el entrenamiento.

### Subproyecto de WebApp

La WebApp se divide en:

- **coleccionesImagenes**: Contiene colecciones de imágenes de ejemplo.
- **config**: Contiene un archivo de configuración `.yaml` con parámetros de solapamiento, márgenes y extensiones.
- **src**: Scripts principales de la lógica de la WebApp, un archivo `requirements.txt` para desplegar la aplicación y una carpeta `.streamlit` con un archivo `.toml` para personalizar la apariencia de la web.
- **static**: Almacena archivos estáticos, como el logo de la app.

### Carpeta de Utilidades

La carpeta **utils** contiene scripts `.py` de utilidad para los subproyectos de preparación y WebApp.


### Entorno de Desarrollo

Se recomienda tener instaladas las siguientes herramientas para reproducir completamente el proyecto:

- **Anaconda**
- **Visual Studio Code**
- **Pulse Secure**
- **WinSCP**
- **Putty**


## Licencia

Este proyecto está licenciado bajo la [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Autores

- **Félix de Miguel Villalba** - *Desarrollador Principal*
- **Daniel Urda Muñoz** - *Tutor*


## Dataset

El dataset utilizado para el desarrollo del proyecto es **HERIDAL** y se puede descargar desde la web de los creadores.
- [Descargar dataset] (([http://ipsar.fesb.unist.hr/HERIDAL%20database.html]))
