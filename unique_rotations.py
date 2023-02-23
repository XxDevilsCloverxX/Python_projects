def unique_with_rotations(A):
    #Get initial value count
    output_arr = set()
    #repeat until all values in A are checked
    while len(A)>0:
        #get the element from the queue
        popped = A.pop(0)
        #try to look for a matching number before running rest of code
        try:
            pos = A.index(popped)
        #if an error was thrown in search, num is unique and can count to the unique values
        except:
            output_arr.add(popped)
            continue
        #run tests on number's radix reversibility for unique points if a match was found
        else:
            occurences=True #a match was found
            #check every remaining value in A starting from the occurence after first found position
            for num in range(pos+1,len(A)):
                #if more than one occurence of a number exists, its reverse will not be unique
                if popped == A[num]:
                    #match found, change occurences because too many matches
                    occurences= False
                    break
            #exiting the loop either means that another occurence was found, or one wasn't. Executes if no second matches were found
            if occurences:
                #attempt a lookup of a reverse of the number:
                popped = str(popped)
                popped = int(popped[::-1])
                try:
                    pos = A.index(popped)
                #if not found in list, the reverse is a unique number
                except:
                    output_arr.add(popped)
                    continue
    uniques = len(output_arr)
    return (uniques, output_arr)

entry = 0
while(True):
    entry+=1
    print(f"Trial entry {entry}:")
    new_list = []
    print("Enter numbers until you are satisfied, then press enter to confirm...")
    while(True):
        #reading numbers
        try:
            user_input = int(input())
        #number wasn't entered
        except:
            break #break the infinite loop
        #integer successfully scanned
        else:
            new_list.append(user_input)
    #execute unique_with_rotations of user input
    output = unique_with_rotations(new_list.copy())
    print(f"{new_list} has {output[0]} unique numbers when you consider the reverse of numbers.\nThose numbers are {output[1]}\n")
    #confirm user intentions:
    flag = input("Enter Y/y or 'Yes' to continue, any other key to exit program")
    if flag.lower() != 'y' and flag.lower() != "yes":
        break
    else:
        print("\n")
