from utils.utilidadesDirectorios import seleccionaDirectorio, obtienePathFicheros
from utils.procesadoXML import getListaBndbox
from utils.entradaSalida import cargaParametrosConfiguracion

import sys

def analizaBndbox(xmlPaths):
    contadorBndbox = 0
    maxAncho = 0
    maxAlto = 0
    minAncho = sys.maxsize
    minAlto = sys.maxsize
    sumAncho = 0
    sumAlto = 0
    for xmlPath in xmlPaths:
        listaBdnbox = getListaBndbox(xmlPath)
        for bndBox in listaBdnbox:
            contadorBndbox += 1
            ancho = bndBox[2] - bndBox[0]
            alto = bndBox[3] - bndBox[1]
            sumAncho += ancho
            sumAlto += alto
            if ancho > maxAncho:
                maxAncho = ancho
            if alto > maxAlto:  
                maxAlto = alto
            if ancho < minAncho:
                minAncho = ancho
            if alto < minAlto:
                minAlto = alto
    
    mediaAncho = sumAncho / contadorBndbox
    mediaAlto = sumAlto / contadorBndbox

    return contadorBndbox, maxAncho, maxAlto, minAncho, minAlto, mediaAncho, mediaAlto

def generaInformeBndbox(ficheroInforme, salidaAnalisis):
    contadorBndbox, maxAncho, maxAlto, minAncho, minAlto, mediaAncho, mediaAlto = salidaAnalisis
    # Abrir el archivo en modo escritura
    with open(ficheroInforme, 'w', encoding='utf-8') as f:
        f.write("Informe de Medidas de Bounding Boxes\n")
        f.write("===============================\n\n")
        
        f.write(f"Número total de Bounding Boxes: {contadorBndbox}\n\n")
        
        f.write("Medidas:\n")
        f.write(f"- Ancho máximo: {maxAncho}\n")
        f.write(f"- Alto máximo: {maxAlto}\n")
        f.write(f"- Ancho mínimo: {minAncho}\n")
        f.write(f"- Alto mínimo: {minAlto}\n")
        f.write(f"- Ancho medio: {mediaAncho}\n")
        f.write(f"- Alto medio: {mediaAlto}\n\n")
        
        f.write("\nFin del Informe")

    print(f"Informe generado correctamente en '{ficheroInforme}'")



def main():
    configuracion = cargaParametrosConfiguracion('../config/parametros.yaml')
    if configuracion == None:
        return
    
    labelsPath = seleccionaDirectorio()
    if labelsPath == None:
        return
    
    xmlPaths = obtienePathFicheros(labelsPath, extensionesPermitidas=['xml'])
    salidaAnalisis = analizaBndbox(xmlPaths)

    ficheroInforme = configuracion['informes']['informeBndbox']
    generaInformeBndbox(ficheroInforme, salidaAnalisis)

if __name__ == '__main__':
    main()