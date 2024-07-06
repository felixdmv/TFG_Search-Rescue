from defusedxml import ElementTree as DET
import xml.etree.ElementTree as ET
import xml.dom.minidom
from utils.utilidadesDirectorios import creaPathDirectorioNivelInferior

def getListaBndbox(xmlPath):
    """
    Retrieves a list of bounding boxes from an XML file.

    Args:
        xmlPath (str): The path to the XML file.

    Returns:
        list: A list of tuples representing the bounding boxes. Each tuple contains the coordinates (xmin, ymin, xmax, ymax).

    """
    listaBndbox = []
    xmlTree = DET.parse(xmlPath) # Se cambia ET por DET para evitar ataques de inyección de entidades externas
    root = xmlTree.getroot()
    objects = root.findall('object')

    for obj in objects:
        bndboxObject = obj.find('bndbox')
        xmin = int(bndboxObject.find('xmin').text)
        xmax = int(bndboxObject.find('xmax').text)
        ymin = int(bndboxObject.find('ymin').text)
        ymax = int(bndboxObject.find('ymax').text)

        listaBndbox.append((xmin, ymin, xmax, ymax))

    return listaBndbox


def createXmlSubimage(imageName, subimageTypePath, listaBndbox, i, j):
    """
    Create an XML file for a subimage with bounding box annotations.

    Args:
        imageName (str): The name of the image.
        subimageTypePath (str): The path to the subimage type.
        listaBndbox (list): A list of bounding box coordinates in the format [xmin, ymin, xmax, ymax].
        i (int): The row index of the subimage.
        j (int): The column index of the subimage.

    Returns:
        None
    """
    xmlSubimage = ET.Element('annotation')
    for bndBox in listaBndbox:
        xmin, ymin, xmax, ymax = bndBox
        object = ET.SubElement(xmlSubimage, 'object')
        ET.SubElement(object, 'name').text = 'human'
        ET.SubElement(object, 'pose').text = 'unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'
        bndBoxSubimage = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndBoxSubimage, 'xmin').text = str(xmin)
        ET.SubElement(bndBoxSubimage, 'xmax').text = str(xmax)
        ET.SubElement(bndBoxSubimage, 'ymin').text = str(ymin)
        ET.SubElement(bndBoxSubimage, 'ymax').text = str(ymax)

    
    pathXmlSubimage = creaPathDirectorioNivelInferior(subimageTypePath, f"{imageName}_{j}_{i}.xml")
    xmlSubimageTree = ET.ElementTree(xmlSubimage)

    xml_str = xml.dom.minidom.parseString(ET.tostring(xmlSubimageTree.getroot())).toprettyxml(indent="\t")
    with open(pathXmlSubimage, "w", encoding='utf-8') as f:
        f.write(xml_str)