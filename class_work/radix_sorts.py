#input is any positive integers
def max_reversed(N, max_swaps):
    swaps = 0
    #convert to a string
    string = str(N)
    #repeats for each number in string
    for i in range(len(string)):
        #strings are char arrays, so get the value of the position of cur max in range of i to end
        cur_range = string[i:]
        print(f"Cur range: {cur_range}")
        #only perform if cur_range is not empty and swap is not at its limit
        if len(cur_range)!=0 and swaps <max_swaps:
            #using cur range: get the max number in cur_range
            max_num = max(cur_range)
            print(f"Max Num of cur_range: {max_num}")
            #get the position of the max number:
            pos = cur_range.index(max_num)
            print(f"Index of max: {pos}")
            #if the position of max number in cur_range is not 0, reverse from i to pos
            if pos!= 0:
                swaps+=1
                flipped_str = cur_range[0:pos+1][::-1]  #returns a reversed string between i and pos
                string = string[0:i] + flipped_str + cur_range[pos+1:]#takes the finished part of string and adds this new flipped part with the remainder on end
                print(f"{string} = OLD: {string[0:i]} + Flipped part: {flipped_str} + Remainder: {string[pos+1:]}\n")
            #if the first number in cur_range is the max, copy cur_range to string
            else:
                string = string[0:i] + cur_range
                print(f"OLD: {string[0:i]} + CUR: {cur_range}\n")
            #print the current attempt of swaps made
            print(f"{swaps}/{max_swaps} swaps attempted:")
    out = int(string)
    return out

entry = 0
while(True):
    entry+=1
    print(f"Trial {entry}")
    try:
        #try these two scans
        user_inp = int(input("Please enter any positive integer:"))
        swap_cap = int(input("How many digits are we allowed to swap?"))
    except:
        #catch any non-number inputs from the user and halt the current run of the program
        print("Error: not a number:")
        continue
    else:
        print(f"The max number that {user_inp} can make by only reversing {swap_cap} numbers is: {max_reversed(user_inp, swap_cap)}")
    finally:
        #this block will always run to confirm the user's intentions
        flag = input("Enter Y/y or 'Yes' to continue, enter any other key to exit program")
        if flag.lower() != 'y' and flag.lower() != "yes":
            break
        else:
            print("\n")

#This program is essentially a radix sort question for any value set passed into the program. The amount allowed to reverse is actually the first x digits that will be sorted in descending order
#for example: passing 1 will create the largest number possible with only 1 reversal between two radix points
