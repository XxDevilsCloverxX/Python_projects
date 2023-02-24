#Function f(x) from assignment
def f(x):
    return (x*x + x +41)
def build_list(n):
    #Start a list to append to
    a_list = []

    #We want values from 1 to n
    #The range function stops at n, so we need n+1
    # and starting from 1
    for i in range(1,n+1):
        a_list.append(f(i))
    return a_list

def isprime(n):
    #1 and less is not prime
    if n<=1:
        return False

    #Create an upper bound to check using sqrt
    upper_bound = int(n**(1/2)) +1

    #Convert the upper bound to an int and go 1 more.
    for i in range(2, upper_bound):
        if (n%i==0):
            return False

    #If all other conditions fail, the number is prime
    return True


#Code from the assignment
mylist = build_list(50)
for k in mylist :
    if isprime(k):
        print("{0} is prime .".format(k))
    else:
        print("{0} is NOT prime.".format(k))

#In SCP i, I tested 3 and 64 in the isprime function where 3 is prime (True) and 64 is not (False)
# The isprime function passed both cases by uncommenting the code below:

#print(isprime(3))
#print(isprime(64))

#In SCP iii, I tested the values 49 and 47, where 49 can be checked below for the isprime Function
# and 47 is part of the printed list from 50 above. Both results of the code and the scp iii matched that 47 is
# prime and 49 is not as in the code below:
# print(isprime(49))
