#node object for Linked list - data organization
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None    #used for next Node
    def print_nodes(self):
        cur = self
        while cur.next!= None:
            cur = cur.next
            print(cur.value," ", end = "")
#takes an unsorted queue of heights, A, and returns the # of rows needed when a new max height is given + Linked list representation of rows
def sort_heights(A):
  #As numbers are read in, if that number is the greatest number, a new row will be added
  B = [] #create an empty list of keys to be scanned
  B_LL = [] #adjacency list that will keep linked lists for printing of data
  while len(A) > 0:
    #pop the first value
    popped = A.pop(0)
    #if the list is empty, add the value:
    if len(B) == 0:
        B.append(popped) #append a first key
        B_LL.append(Node(popped)) #append a first node object
    else:
        #if the popped value is greater than or equal to cur max of list, add the popped val to the lists
        if popped >=max(B):
            B.append(popped)   #add key
            B_LL.append(Node(popped)) #add node obj
        else:
            #get the lowest key that popped is less than
            for key in B:
                if key>popped:
                    pos = B.index(key)
                    #go to LL index
                    cur = B_LL[pos]
                    while(cur.next!=None):
                        cur = cur.next #go to next node object
                    cur.next = Node(popped)
                    break #stop at first insertion
  min_rows = len(B)
  #return a tuple of min rows and completed adjacency list of how the rows would look
  return min_rows, B_LL

#create a loop that asks users to input values
entry = 0
while(True):
    entry+=1
    print(f"Trial entry {entry}:")
    new_list = []
    print("Enter heights as they are queued, then press enter to confirm...")
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
    #run a sort_heights on the user-generated-list of heights
    output = sort_heights(new_list.copy())
    print(f"{new_list} would require {output[0]} a minimum of rows in this order.")
    flag = input("Would you like to see easiest organization of these rows? Y/n or 'yes'.")
    if flag.lower() == 'y' or flag.lower()=="yes":
        for head in output[1]:
            #end="" keeps row on same line
            print(head.value,": [", end = "")
            #print linked list values
            head.print_nodes()
            print("]")
    flag = input("Would you like to continue? Y/n or 'yes'.")
    if flag.lower() != 'y' and flag.lower()!="yes":
        print("Thank you for using the program!")
        break#exit the program
