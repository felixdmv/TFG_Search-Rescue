## Dataset de subimágenes

El dataset de subimágenes contiene las mismas imágenes que el dataset 'Heridal', pero estas están troceadas más pequeñas en 340x340 píxeles. 
Estas subimágenes cuadradas se utilizan para entrenar un modelo de clasificación que identificará la presencia de humanos en las imágenes. 

Además, se incluye para cada subimagen su correspondiente .xml con el mismo nombre, el cual incluye la presencia o no de humanos en coordenadas (x,y)

El dataset se divide en 13 tipos de terreno distintos, y cada uno de ellos contiene un .csv que detalla si una imagen tiene o no humano y su caja correspondiente.
Las cajas nos servirán para hacer una validación cruzada en el momento de hacer el análisis.

Debido a limitaciones de espacio, se incluye el recorte de las imágenes en One Drive.

Puedes descargar este dataset desde el siguiente enlace:

- [Descargar dataset de subimágenes 'Heridal'](https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/fdv1004_alu_ubu_es/ERVMKBhG6j9MlwgU7YR4LP8Bcz50GMSapRGofSmyh-95uA?e=ccG4JH)



