import pymysql
import os
from glob import glob
import BytesIO
import argparse
from easyocr import Reader

directory = os.path.dirname(os.path.realpath(__file__))  #get the directory of this file
directory = os.path.join(directory, "outputs")    #get the directory of the license plates

def upload_to_database(directory, connection):
        i = -1  #initialize the default index for printing
        print("Uploading results to database...")
        #loop over the buffer
        for file in glob(directory + "/*.jpeg"):
            if plate not in self.shift_buffer.keys():
                try:
                    with self.connection0.cursor() as cursor:
                        #create the sql query
                        sql = "INSERT INTO `licenses` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                        #execute the query
                        cursor.execute(sql, (plate, self.buffer[plate]))
                        #commit the changes
                        self.connection0.commit()
                
                    #upload the results to the database
                    print(f"'{plate}' uploaded to database!")
                except pymysql.err.OperationalError as e:
                    print(f"Error uploading '{plate}' to database: {e}")
                    print("Pushing to alternate DB")
                    #Upload to mariadb
                    with self.connection1.cursor() as cursor:
                        #create the sql query
                        sql = "INSERT INTO `licenseplates` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                        #execute the query
                        cursor.execute(sql, (plate, self.buffer[plate]))
                        #commit the changes
                        self.connection1.commit()
    else:
        print("Printing results...")
        #loop over the buffer
        for i, plate in enumerate(self.buffer.keys()):
            if plate not in self.shift_buffer.keys():
                print(f"{plate}")  #print the plate

    print(f"{i+1} results captured + pushed!")

    #update the time buffer
    self.shift_buffer.clear()              #clear the time buffer before updating
    self.shift_buffer.update(self.buffer)  #update the time buffer
    
    #clear the buffer
    self.buffer.clear()