import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'): # glob.glob() returns a list of files that match the pattern
        tree = ET.parse(xml_file)            # parse the xml file   
        root = tree.getroot()             # get the root of the xml file
        for member in root.findall('object'):   # find all the objects in the xml file
            bbx = member.find('bndbox')  # find the bounding box of the object
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text    # find the name of the object

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),    # find the width of the image
                     int(root.find('size')[1].text),    # find the height of the image
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    path = os.path.join(os.path.dirname(__file__), "images")
    datasets = ['training_data', 'testing_data']    # the folder name of the images and labels
    for ds in datasets:
        image_path = os.path.join(path, ds)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('labels_{}.csv'.format(ds), index=None)
        print('Successfully converted xml to csv.')


main()

"""
This is a pascal voc to csv converter.
"""