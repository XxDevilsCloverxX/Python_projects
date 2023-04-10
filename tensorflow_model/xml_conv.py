import os
import xml.etree.ElementTree as ET

# Set the directory path
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        # Parse the XML file
        tree = ET.parse(os.path.join(directory, filename))
        root = tree.getroot()

        # Find the filename tag and update the extension
        for filename_tag in root.iter('filename'):
            filename_tag.text = filename_tag.text.replace('.png', '.jpeg')

        # Write the updated XML to a new file with the same name
        tree.write(os.path.join(directory, filename))

print('All XML files updated.')
