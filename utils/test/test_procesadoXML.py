import os
import pytest
from utils.procesadoXML import getListaBndbox, createXmlSubimage

@pytest.fixture
def setup_test_files():
    # Directory for test files
    test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    

    # Create test XML files with corrected content
    test_xml_1 = os.path.join(test_files_dir, "test1.xml")
    with open(test_xml_1, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" ?>
<annotation>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>100</xmin>
         <xmax>200</xmax>
         <ymin>150</ymin>
         <ymax>250</ymax>
      </bndbox>
   </object>
</annotation>''')

    test_xml_2 = os.path.join(test_files_dir, "test2.xml")
    with open(test_xml_2, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" ?>
<annotation>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>300</xmin>
         <xmax>400</xmax>
         <ymin>350</ymin>
         <ymax>450</ymax>
      </bndbox>
   </object>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>500</xmin>
         <xmax>600</xmax>
         <ymin>550</ymin>
         <ymax>650</ymax>
      </bndbox>
   </object>
</annotation>''')

    return test_files_dir

def test_getListaBndbox(setup_test_files):
    test_xml_1 = os.path.join(setup_test_files, "test1.xml")
    test_xml_2 = os.path.join(setup_test_files, "test2.xml")

    lista_bndbox_1 = getListaBndbox(test_xml_1)
    assert lista_bndbox_1 == [(100, 150, 200, 250)]

    lista_bndbox_2 = getListaBndbox(test_xml_2)
    assert lista_bndbox_2 == [(300, 350, 400, 450), (500, 550, 600, 650)]

def test_createXmlSubimage(setup_test_files):
   test_xml_1 = os.path.join(setup_test_files, "test1.xml")
   lista_bndbox_1 = getListaBndbox(test_xml_1)

   # Path where the new XML will be created
   subimageTypePath = os.path.join(setup_test_files, "subimages")
   os.makedirs(subimageTypePath, exist_ok=True)

   createXmlSubimage("test_image", subimageTypePath, lista_bndbox_1, 1, 1) 

   # Verify the new XML file was created
   expected_xml_path = os.path.join(subimageTypePath, "test_image_1_1.xml")
   assert os.path.exists(expected_xml_path)
   test_xml_1 = os.path.join(setup_test_files, "test1.xml")
   with open(test_xml_1, 'r') as f:
      expected_content = f.readlines()
            
   with open(expected_xml_path, 'r') as f:
      actual_content = f.readlines()

   assert len(expected_content) == len(actual_content), f'el número de líneas en {expected_xml_path} no coincide con el número de líneas en {actual_content}'
   for i in range(len(expected_content)):
      assert actual_content[i].strip() == expected_content[i].strip(), f'la línea {expected_content[i]} difiere de {actual_content[i]}'
   
   

if __name__ == "__main__":
    pytest.main()
