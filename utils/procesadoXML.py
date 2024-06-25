from defusedxml import ElementTree as ET
import xml.dom.minidom
from utils.utilidadesDirectorios import creaPathDirectorioNivelInferior

def getListaBndbox(xmlPath):
    listaBndbox = []

    xmlTree = ET.parse(xmlPath)
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
    xmlSubimage = ET.Element('annotation')
    for bndBox in listaBndbox:
        xmin, ymin, xmax, ymax = bndBox
        obj = ET.SubElement(xmlSubimage, 'object')
        ET.SubElement(obj, 'name').text = 'human'
        ET.SubElement(obj, 'pose').text = 'unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndBoxSubimage = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndBoxSubimage, 'xmin').text = str(xmin)
        ET.SubElement(bndBoxSubimage, 'xmax').text = str(xmax)
        ET.SubElement(bndBoxSubimage, 'ymin').text = str(ymin)
        ET.SubElement(bndBoxSubimage, 'ymax').text = str(ymax)

    pathXmlSubimage = creaPathDirectorioNivelInferior(subimageTypePath, f"{imageName}_{j}_{i}.xml")
    xmlSubimageTree = ET.ElementTree(xmlSubimage)

    xml_str = xml.dom.minidom.parseString(ET.tostring(xmlSubimageTree.getroot())).toprettyxml(indent="\t")
    with open(pathXmlSubimage, "w") as f:
        f.write(xml_str)

'''
import xml.etree.ElementTree as ET
import xml.dom.minidom
from utils.utilidadesDirectorios import creaPathDirectorioNivelInferior

def getListaBndbox(xmlPath):
    listaBndbox = []

    xmlTree = ET.parse(xmlPath)
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
    with open(pathXmlSubimage, "w") as f:
        f.write(xml_str)
    '''