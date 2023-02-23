import pandas as pd

mycsv = open("msg.csv", "w")
mycsv.write("Date,Time,Author,Message\n")

with open("temp.txt", 'r') as f:
    for line in f.readlines():
        #clean line by removing brackets and uneeded apostrophies. next, split by content ender
        splitter = line.replace("[", "").replace("'", "").strip().split("]")
        splitter.pop(-1)    #remove unneccesarry block, remove this adds a "" entry to list

        csvblock = []  #bad practice, but late and need to finish this lmao

        for item in splitter:
            csvline = ","
            #you have to split by spaces to check for date times
            prep = item.split()
            try:
                #will throw an error if date-time format is not matched with string format
                date = "%d/%d/%d"%(int(prep[0][0:2]), int(prep[0][3:5]), int(prep[0][6:]))
                time = "%d:%d"%(int(prep[1][0:2]), int(prep[1][3:]))
            except:
                #Exception: no date was given so we append the message with the assumption of same author in quick succession
                csvblock[-1]+= " " + " ".join(prep[:]).replace(",", "")
            else:
                #if there is a new date, there is an author in item that is accesed from the - to the : create a new line in the csv
                index = prep.index("-") +1
                findex = 0
                #look for author colon
                for num in range(index, len(prep)):
                    if ":" in prep[num]:
                        findex = num+1
                        break
                
                #get author data (Reason for loop above was for authors with one name, no last, no first, middles, etc.)
                author = " ".join(prep[index:findex]).replace(":", "")  #remove the colon
                message = " ".join(prep[findex:]).replace(",","") #everything after the author block
                csvline= csvline.join((date,time,author, message))
                #append csvline to the csvblock list
                csvblock.append(csvline)
                
        #added this loop to properly add messages from same user to the table
        for towrite in csvblock:
            mycsv.write(towrite+"\n") #write the latest line to the csv
f.close()
mycsv.close()

#now create the dataframe:
with open("msg.csv", "r") as mycsv:
    df = pd.read_csv(mycsv)
    print(df)