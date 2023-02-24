# Understand the problem:
    #The problem examples are worked out by hand on the attached PDF
    #With the submission. However, I used the following:
    # A = 2, n = 3 to get y_4 = 1.41421569.
    # A = 7, n = 2 to get y_3 = 2.87500000.
    # A = 3, n = 4 to get y_5 = 1.73205081.

## Devlop an Approach:
    #On PDF, will include brief here:
    #Scan a user input for A and n with appropriate data type
    #Compute the current element in the sequence
    #Set the previous element to computed element so that the
    #'new previous element' can be used in next computation
    #print the final element at the termination of the loop.
### Test Approach by hand:
    #The approach by hand is on attached PDF and verifies part I

#### Code it:

#Reading inputs from Users.
A = float(input("Please enter a real number greater than or equal to 1 for A.\n"))
if A<1:
    print("Fatal Error: A < 1.")
    quit()
n = int(input("Please enter an integer greater than or equal to 1.\n"))
if n<1:
    print("Fatal Error: n < 1.")
    quit()
y_previous = 1 #initialize the first term in the series. This value precedes all values in sequence
compare = 1 #index var for while loop
while(compare < n+1):
    y_np1 = 0.5*(y_previous + (A / y_previous)) #Compute the preceding step
    y_previous = y_np1 #Set the computed step to previous so that it becomes the new substition.
    compare+=1
print(f"For the element following term {n} of the sequence: y_{compare} = {y_np1:8.8f}") #fstrings allow formatting with this syntax :)
