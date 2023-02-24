#Code copied from the assignment:
def gcd(m,n):
    if n==0:
        return m
    return gcd(n, m%n)

#The function that does the same thing as gcd, except that it counts operations
def gcd_divops(m,n):
    #Return 0 division operation if n is 0
    if n==0:
        return 0
    #Return 1 operation for the step, and calculate again using a flipped order
    #and m%n to get the remainder.
    return(1+gcd_divops(n, m%n))

#Code to test GCD between numbers and display how many division operations were needed.
a = 1644; b=1264
print(f"{gcd(a, b)} is greatest common divisor of {a} and {b}.")
print(f"{gcd_divops(a,b)} is the division operations.\n")

#Compute the division operations of every combination of i and j for a square space of area 10^2, 100^2, 1000^2
sum=0
for N in [10,100,1000]:
    #line_num = 1
    for i in range(1,N+1):
        for j in range(1,N+1):
            sum += gcd_divops(i,j)
            #print(f"{line_num}: gcd_divops({i},{j}): Current sum: {sum} current N: {N} Divops: {gcd_divops(i, j)}")
            #line_num +=1
    print(f"Average of the division operations of range [{1},{N}]: {sum/(N*N)}")
    sum = 0

#This was to help with proofs on Part 3
#b = 4
#for i in range(1,4):
#    print(f"For a = {i}, b = {b} there are {gcd_divops(i,b)} operations")

#   Comparison with bound calculated from part 3:
#   2*log_2(10) ->(apx) 6.64 we got 2.21 with the program
#   2*log_2(100) ->(apx) 13.29 we got 4.0047 with the program
#   2*log_2(1000) ->(apx) 19.93 we got 5.933071 with the program

#   So for the comparison, we are well within the bounds, and the distance from the upper bound keeps increasing.
