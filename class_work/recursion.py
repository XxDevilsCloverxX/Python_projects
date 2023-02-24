#We will not be running zero through this function
def C3n1(n):
    #Using // to return an integer instead of float
    if (n%2 == 0):
        return (n//2)
    else:
        return ((3*n+1)//2)

#Recursive function that solves the problem
def Csequence(n):
    #Simple case that n is 1:
    if n==1:
        return [1]
    #If complex, use the reducing step to concatenate the list, starting from first n
    else:
        return [n] + Csequence(C3n1(n))

m = int( input ("Enter a positive integer m:"))
seq = Csequence(m)
print ("The resulting sequence is: {0} ".format(seq))
print ("It has length {0} ".format(len(seq)))

#Tested values from SCP:
    #input   outputs
    #  12       [12,6,3,5,8,4,2,1] and len 8
    #  7        [7,11,17,26,13,20,10,5,8,4,2,1] and len 12
    #  8        [8,4,2,1], len 4

#First 2 cases matched SCP i
#Last case matched SCP iii using SCP ii ruleset
#which is verifyable in scp i examples
