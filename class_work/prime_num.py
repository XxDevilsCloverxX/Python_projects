def get_roots(p):
    a_seq = [] #initialize an empty list to append to
    i = 0 #index counter
    while(i<p): #condition where to stop checking [0,p)
        n = i*i + 3*i+5 #set shorthand variable
        if(n%p == 0): #if the value is divisible by the prime number
            a_seq.append(i) #add the integer this occurs at to the list.
    #after all values are found, return the list
        i+=1 #increment i at the end
    return a_seq

#Code from the exam
q = int(input("Please enter a prime number: "))
solns = get_roots(q)
print ("The roots modulo {} are: {} ".format(q, solns))


#In scp 3, tested p = 5 and got output: The roots modulo 5 are: [0, 2]
#This matches with my result in scp 3.

#####For scp i, I tested p = 3 and p = 2; their results were [1,2] and [], respectively.
#### The outputs from the program are:
### The roots modulo 3 are: [1, 2].
## and The roots modulo 2 are: [].
#This aligns with scp i in my pdf submission
