from PIL import ImageDraw
import seaborn as sns

def dibujaRectangulos(image, listaRectangulos, ancho=4, color='red'):
    """
    Draws rectangles on an image.

    Args:
        image (PIL.Image.Image): The image on which to draw the rectangles.
        listaRectangulos (list): A list of tuples representing the coordinates of the rectangles.
            Each tuple should contain four values: left, upper, right, lower.
        ancho (int, optional): The width of the lines used to draw the rectangles. Defaults to 4.
        color (str, optional): The color of the lines used to draw the rectangles. Defaults to 'red'.

    Returns:
        PIL.Image.Image: A copy of the input image with the rectangles drawn on it.
    """
    imagenCopia = image.copy()  # Para conservar la imagen original 
    for left, upper, right, lower in listaRectangulos:
        draw = ImageDraw.Draw(imagenCopia)
        draw.line((left, upper, left, lower), fill=color, width=ancho)
        draw.line((right, upper, right, lower), fill=color, width=ancho)
        draw.line((left, upper, right, upper), fill=color, width=ancho)
        draw.line((left, lower, right, lower), fill=color, width=ancho)
    return imagenCopia

def creaListaRectangulosConIndices(imageSize, subimageSize, overlap, margins):
    width, height = imageSize
    subimageSize_x, subimageSize_y = subimageSize
    overlap_x, overlap_y = overlap
    margin_x, margin_y = margins

    listaRectangulosConIndices = []

    col = 0
    for left in range(0, width, subimageSize_x - overlap_x):
        fil = 0
        right = left + subimageSize_x
        if right <= width:
            for upper in range(0, height, subimageSize_y  - overlap_y):
                lower = upper + subimageSize_y 
                if lower <= height:
                    listaRectangulosConIndices.append(((left, upper, right, lower), (col, fil)))
                    fil +=1
            if height - upper >= margin_y:
                listaRectangulosConIndices.append(((left, height- subimageSize_y, right, height), (col, fil)))
                fil += 1
            col += 1
   
    if width - left >= margin_x:
        for upper in range(0, height, subimageSize_y  - overlap_y):
            lower = upper + subimageSize_y 
            if lower <= height:
                listaRectangulosConIndices.append(((width - subimageSize_x, upper, width, lower), (col, fil)))
                fil += 1
        if height - upper >= margin_y:
            listaRectangulosConIndices.append(((width - subimageSize_x, height- subimageSize_y, width, height), (col, fil)))
            fil += 1
    
    return listaRectangulosConIndices

def creaListaRectangulos(imageSize, subimageSize, overlap, margins):
    width, height = imageSize
    subimageSize_x, subimageSize_y = subimageSize
    overlap_x, overlap_y = overlap
    margin_x, margin_y = margins

    listaRectangulos = []
    for left in range(0, width, subimageSize_x - overlap_x):
        right = left + subimageSize_x
        if right <= width:
            for upper in range(0, height, subimageSize_y  - overlap_y):
                lower = upper + subimageSize_y 
                if lower <= height:
                    listaRectangulos.append((left, upper, right, lower))
            if height - upper >= margin_y:
                listaRectangulos.append((left, height- subimageSize_y, right, height))
                
   
    if width - left >= margin_x:
        for upper in range(0, height, subimageSize_y  - overlap_y):
            lower = upper + subimageSize_y 
            if lower <= height:
                listaRectangulos.append((width - subimageSize_x, upper, width, lower))
                
        if height - upper >= margin_y:
            listaRectangulos.append((width - subimageSize_x, height- subimageSize_y, width, height))
           
    
    return listaRectangulos

def creaRectangulos(tamImOriginal, tamSubimagen, solapamiento):
    """
    Creates a list of rectangles that divide an image into subimages.

    Args:
        tamImOriginal (tuple): The size of the original image (width, height).
        tamSubimagen (tuple): The size of the subimage (subimage_size_x, subimage_size_y).
        solapamiento (tuple): The amount of overlap between subimages (overlap_x, overlap_y).

    Returns:
        tuple: A tuple containing the list of rectangles, the number of rows, and the number of columns.
    """
    width, height = tamImOriginal
    subimage_size_x, subimage_size_y = tamSubimagen
    overlap_x, overlap_y = solapamiento

    listaRectangulos = []
    numImagenes = 0
    col = 0
    for left in range(0, width, subimage_size_x - overlap_x):
        right = left + subimage_size_x
        if right <= width:
            col += 1
            for upper in range(0, height, subimage_size_y - overlap_y):
                lower = upper + subimage_size_y
                if lower <= height:
                    listaRectangulos.append((left, upper, right, lower))
                    numImagenes += 1
    return listaRectangulos, numImagenes//col, col




def dibujaRejilla(image, listaRectangulos, ancho=2, color='black'):
    """
    Draws a grid on the given image using the provided list of rectangles.

    Args:
        image (PIL.Image.Image): The image on which the grid will be drawn.
        listaRectangulos (list): A list of rectangles represented as tuples (left, upper, right, lower).
        ancho (int, optional): The width of the grid lines. Defaults to 2.
        color (str, optional): The color of the grid lines. Defaults to 'black'.

    Returns:
        PIL.Image.Image: The image with the grid drawn on it.
    """
    imagenCopia = image.copy()  # Para conservar la imagen original 
    draw = ImageDraw.Draw(imagenCopia)
    for left, upper, right, lower in listaRectangulos:
        draw.line((left, upper, left, lower), fill=color, width=ancho)
        draw.line((right, upper, right, lower), fill=color, width=ancho)
        draw.line((left, upper, right, upper), fill=color, width=ancho)
        draw.line((left, lower, right, lower), fill=color, width=ancho)
    return imagenCopia


def mapaCalor(predicciones, fil, col):
    """
    Generate a heatmap plot based on the given predictions.

    Args:
        predicciones (numpy.ndarray): The predictions to be visualized.
        fil (int): The number of rows in the heatmap.
        col (int): The number of columns in the heatmap.

    Returns:
        matplotlib.figure.Figure: The generated heatmap plot.

    """
    matrizPredicciones = predicciones.reshape(col, fil).T
    plot = sns.heatmap(matrizPredicciones, annot_kws={"fontsize":4}, cbar_kws={"shrink": 0.75}, square=True, vmin=0.0, vmax=1.0, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, xticklabels=False, yticklabels=False)
    plot.set_title('Mapa de Calor de Predicciones', fontsize=10)
    cax = plot.figure.axes[-1]
    cax.tick_params(labelsize=6)  # Tamaño de texto en barra de color
    return plot.get_figure()