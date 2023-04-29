import pymysql
import os
from glob import glob
from io import BytesIO
import argparse
from easyocr import Reader
import cv2
from PIL import Image
import pandas as pd

def upload_to_database(connection, ocrdict):
        i = -1  #initialize the default index for printing
        print("Uploading results to database...")
        #loop over the buffer
        try:
            #get the license plate from the file name
            for i, plate in enumerate(ocrdict.keys()):
                with connection.cursor() as cursor:
                    #create the sql query
                    sql = "INSERT INTO `licenses` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                    #execute the query
                    cursor.execute(sql, (plate, ocrdict[plate]))
                    #commit the changes
                    connection.commit()
                
                    #upload the results to the database
                    print(f"'{plate}' uploaded to database!")
                
        except pymysql.err.OperationalError as e:
            print(f"Error uploading '{plate}' to database: {e}")
        print(f"Uploaded {i+1} results to database!")

def get_ocr_results(image, filter):
     #create the reader
    reader = Reader(['en'])
    #read the image
    results = reader.readtext(image, detail=0)
    # join the words together
    plate_text = "".join(results)
    plate_text = [char.upper() for char in plate_text if char.isalnum()]    #remove special characters
    plate_text = "".join(plate_text)    #join the characters together
    
    # filter out the states
    if filter is not None:
        for state in filter:
            if state.upper() in plate_text:
                plate_text = plate_text.replace(state.upper(), '')  #remove the state from the plate text

    # return a max of 7 characters
    if len(plate_text) > 7:
        plate_text = plate_text[:7]

    elif len(plate_text) < 4:
        plate_text = "" #return an empty string if the plate is too small

    print(f"Plate text: {plate_text}")

    return plate_text    #set the plate text
        
def get_ocr_results_from_dir(directory, filter):
    ocrdict = {}    #initialize the dictionary
    stream = BytesIO()  #initialize the stream
    #loop over the files in the directory
    for file in glob(directory + "/*.jpeg"):
        #open the file
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop = Image.open(file)
        #get the plate text
        plate_text = get_ocr_results(image, filter)
        #add the plate text to the dictionary
        if plate_text != "":
            #use the stream to write a bblob
            crop.save(stream, format='JPEG')
            ocrdict.update({plate_text: stream.getvalue()})
            stream.seek(0)  #reset the stream
            stream.truncate()   #reset the stream

    return ocrdict

if __name__ == '__main__':
    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filterpath", type=str,
                    help="Path to the csv filter model")
    args = vars(ap.parse_args())

    # get the filter
    if args["filterpath"] is not None:
        filter = pd.read_csv(args["filterpath"])["State"].tolist()
    
    filter = [state.upper() for state in filter]  #convert the states to uppercase

    try:    
        #define the sql connections
        conn = pymysql.connect(
        host = "195.179.237.153",   #ip address of the database
        user = "u376552790_projectlabdb",   #username of the database
        password = "Ltg5075@",  #password of the database
        db = "u376552790_projectlabdb"  #name of the database
        )
    except Exception as e:
        print(e, "Database connection failed!")
        conn = None
    try:
        conn2 = pymysql.connect(
        host = 'localhost',
        user = 'clover',
        password = 'Silas',
        db = 'platedb'
        )
    except Exception as e:
        print(e, "Database connection failed!")
        conn2 = None

    directory = os.path.dirname(os.path.realpath(__file__))  #get the directory of this file
    directory = os.path.join(directory, "outputs")    #get the directory of the license plates

    #get the ocr results
    ocrdict = get_ocr_results_from_dir(directory, filter)

    #upload the results to the database
    if conn is not None:
        upload_to_database(conn, ocrdict)
    if conn2 is not None:
        upload_to_database(conn2, ocrdict)

    #close the connections
    if conn is not None:
        conn.close()
    if conn2 is not None:
        conn2.close()

    print("Done!")