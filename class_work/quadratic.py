# This program is a simple quadratic formula calculator

A = float(input("Please enter A.")) #A,B,C will be entered
B = float(input("Please enter B.")) #For use within
C = float(input("Please enter C.")) #The Quadratic Formula

x1 = (-B + (B**2 - 4*A*C)**(1/2))/(2*A) #Solution x1 of the Quadratic Formula
x2 = (-B - (B**2 - 4*A*C)**(1/2))/(2*A) #Solution x2 of the Quadratic Formula

print (x1,x2)

#When A = 1.0, B = 3.0, C = -4.0, x1 = -4.0, x2 = 1.0.
#When A = 2, B = 7, C = 3, x1 = -0.5, x2 = -3.0.

print("If I were to divide by 0, A=0, the program would throw a ZeroDivisionError.")
