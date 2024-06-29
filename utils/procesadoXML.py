import os
from defusedxml.ElementTree import parse, tostring, Element, SubElement, ElementTree
#from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring
import xml.dom.minidom

from utils.utilidadesDirectorios import creaPathDirectorioNivelInferior

def getListaBndbox(xmlPath):
    listaBndbox = []

    xmlTree = parse(xmlPath)
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
    Creates a new XML file with a subset of bounding boxes.
    """
    xmlSubimage = Element('annotation')
    for bndbox in listaBndbox:
        obj = SubElement(xmlSubimage, 'object')
        name = SubElement(obj, 'name')
        name.text = 'human'
        pose = SubElement(obj, 'pose')
        pose.text = 'unspecified'
        truncated = SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox_elem = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox_elem, 'xmin')
        xmin.text = str(bndbox[0])
        ymin = SubElement(bndbox_elem, 'ymin')
        ymin.text = str(bndbox[1])
        xmax = SubElement(bndbox_elem, 'xmax')
        xmax.text = str(bndbox[2])
        ymax = SubElement(bndbox_elem, 'ymax')
        ymax.text = str(bndbox[3])
    
    pathXmlSubimage = creaPathDirectorioNivelInferior(subimageTypePath, f"{imageName}_{j}_{i}.xml")
    xmlSubimageTree = ElementTree(xmlSubimage)

    xml_str = xml.dom.minidom.parseString(tostring(xmlSubimageTree.getroot())).toprettyxml(indent="\t")
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