##Function that returns number of solutions of x^2+y^2 = Z
#while printing those solutions.
def count_sum_of_squares(Z):

    x=1 #first value
    y=1 #second value
    solutions = 0 #number of solutions

    ### Continue the loop until both
    ## x^2+y^2 is greater or equal to Z AND y^2 >Z.
    # translated into python backwards from pdf submission.
    while(((x*x)+(y*y) < Z) or ((y*y) <= Z)):

        ##check if x^2 + y^2 is a solution
        if((x*x + y*y)==Z): ##Used x*x to optimize on bytecode level
            #confirm easy case where we can exit the loop.
            print(f"Solution of {x}^2 + {y}^2 = {Z}")
            solutions +=1

        #if x^2 becomes = Z or bigger, the loop cannot have a possible
        ##combination for a solution x^2+y^2 = Z
        if(x*x >= Z):
            #increment y
            y+=1
            #reset x
            x=1
        #Increment normally, use else to prevent over increments
        else:
            x+=1
    return solutions


A=50
while A >= 1:
    numsq = count_sum_of_squares(A)
    print ("{0} can be written as a sum of squares in {1} ways.".format(A,numsq))
    A = int(input("Enter a positive integer : "))
