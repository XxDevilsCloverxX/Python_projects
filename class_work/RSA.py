import random as rand
#returns b**a % N
def powmod(b,a,N):
    r = 1
    beta = b
    alpha = a
    while alpha>0:
        #Check if alpha is odd
        if alpha % 2 ==1:
            r = r * beta % N
        #use integer division
        alpha = alpha//2
        beta  = beta**2 % N
    return r

#use powmod to print all integers between 3 and 100 that satisfy 2**(n-1)%n = 1%n
#for n in range(3,100+1):
#    if powmod(2,n-1,n) == 1 % n:
#        print(n)

#q must be odd and t>=1
def mr_single_test(N, q, t):
    #create a list of integers up to N-1
    #int_list = [x for x in range(2, N)]
    #Create a list that contain roots in order b**q mod N, b**2q mod N, b**(2**2)*q mod N...b**(2**t)* q mod N
    roots = []
    #b = rand.choice(int_list)
    b=2
    #use powmod to get first number in a root list using a random integer from int_list
    out = powmod(b, q,N)
    #add this output to the roots list:
    roots.append(out)
    #apppend all the squared powers t times to the list:
    for i in range(t+1):
        #get the previous number in the list
        y = roots[i]
        #square the term:
        y = y**2 % N #reduce using mod N
        #add y to the list of roots
        roots.append(y)
    #If no values are 1, then the N is proven to be composite:
    flag = True
    for i in roots:
        #if any value is 1, the N may be prime
        if i==1:
            flag = False
    if flag: #only runs if there was no 1
        return False
    #otherwise, e is the least nonnegative integer for which b**(2**e)*q mod N ==1
    e = roots.index(1) #get first instance of 1
    #if e was 0
    if e==0:
        return True
    #check this condition
    if e>0 and powmod(b,(2**(e-1))*q, N) == N-1:
        return True
    #no returns at this point means that N is composite
    return False

#test if N is prime, N and k are positive integers
def mr_test(N, k):
    #1 and even numbers are not prime
    if N==1 or N%2==0:
        return False
    #find a t and q such that N-1 == (2**t)*q AND q is odd
    t=0
    #this is finding a multiple of powers of 2
    q = N-1
    while(q%2 == 0):
        #divide case by 2:
        q = q //2
        #increase t
        t+=1
    #call mr_single_test up to k times, returning False should it ever return False for composite numbers
    for i in range(k):
        if(mr_single_test(N, q, t) == False):
            return False
    #Otherwise, it is still possible that N is composite
    #It can be proven that the probability of making such an unlucky choice once
    #is no more than 1/4, and the choices were made independently; so the chance that N
    #is composite and we made k unlucky choices is not more than 1/4**k.
    return True

#find and return next smallest prime p>=n using the function mr_test; using k=20
def nextprime(n):
    #if n is even, make it odd
    if n%2==0:
        n+=1
    #test until mr_test returns prime
    while(mr_test(n,20)== False):
        n+=2
    return n #where n is the next prime

#Final starts here:
    #function copied from inversemodn.py
def inverse_mod_N(a, N):
    # Return the multiplicative inverse of a modulo N, if it exists,
    # or None if a is not invertible modulo N.
    def gcd_ext(x,y):
        # Extended Euclidean Algorithm.
        if y[2]==0:
            return x
        d = x[2]//y[2]
        return gcd_ext(y, (x[0]-d*y[0], x[1]-d*y[1], x[2]-d*y[2]))

    q = gcd_ext( (1,0,a), (0,1,N))
    if q[2]==1:
        return (q[0]%N)
    return None

    #generate a list of integers in range (2**511, 2**512 -1) - note range ends 1 shy of upper bound

#call random.seed(1) for results identical to assignment
rand.seed(1)
    #compute nextprime of 2 integers chosen randomly from 2**511 to 2**512
a = rand.randint(2**511, 2**512 -1)   #random int a
b = rand.randint(2**511, 2**512 -1)   #random int b
p = nextprime(a)
q = nextprime(b)
    #find least positive integer for which inverse_mod_N does not return none
    #e will be passed to first arg starting from 65537
e = 65537
while(inverse_mod_N(e, (p-1)*(q-1)) == None):
    e+=1 #go to next e
#Exiting loop means there is an e >= 65537 for which the function did not return None
d = inverse_mod_N(e, (p-1)*(q-1))   #call one more time using the last parameters of loop to get d
N = p*q
#set private and public keys, they are tuples according to the assignment
public_key = (N,e)
private_key = (N,d)
#print the keys
print(f"Public Key: {public_key}\n")
print(f"Private Key: {private_key}\n")

#encrypt this message
m = 1234567890123456789012345678901234567890

#using the public key, specifically e, encrypt the message using y = m**e mod N
    #e = public_key[1], N = public_key[0]
y = powmod(m, public_key[1], public_key[0])

#print the encrypted message as a base 10 int
print(f"Encrypted Message: {y}\n")

#after "sending y", decrypt the message and print as a base 10 int: m = y**d mod N
    #d = private_key[1], N = private_key[0]
m = powmod(y, private_key[1], private_key[0])

print(f"Decrypted Message: {m}\n")

    #Below is a point made in the assignment I found interesting
#Note: this is actually very insecure, even without setting the seed to 1, because it is
#possible in most cases to narrow down the list of possible ‘seeds’ that the random module
#might have used to a small range; in that case, an attacker can ‘factor’ the N by directly
#trying to produce primes in the same way, and checking whether they divide your N . This
#is a crucial pitfall to be avoided when using this system for real!

#Investigate how Python would seed the PRNG if we didn't seed it ourselves.
#Then estimate how long it takes to generate a prime number and you can ballpark
#how long it would take to crack a key if you knew the day/year/decade in which it
#was created.
