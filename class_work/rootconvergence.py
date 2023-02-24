import math as m
#Define Testing Functions and Derivatives here

##From Problem 1
def f(t):
    return t**2 - 7
def df(t):
    return 2*t
##From SCP 3 of PDF
def y(t):
    return t**2 - 4
def dy(t):
    return 2*t
##From Problem 2
def h(x):
    return x**5 -5*x**3+4*x+1
def dh(x):
    return 5*x**4-15*x**2+4
##From Problem 3
def G(x):
    return (m.exp(-2*x**2)-0.5-0.5*x)
def dG(x):
    return (-4*x*m.exp(-2*x**2) - 0.5)
# Defining the Newton Function from Problem 1
def Newton(f, df, x_0, epsilon):
    x_n = x_0
    trial = 0
    while (trial < 30):
        fx_n = f(x_n)
        dfx_n = df(x_n)
        if (abs(fx_n) < epsilon):
            return x_n
        if (abs(dfx_n) < epsilon):
            print(f"Error converging from x_0 = {x_0}")
            return None
        x_np1 = x_n - (fx_n / dfx_n)
        x_n = x_np1
        trial +=1
    print(f"Exceeded trial limit of 30 from x_0 = {x_0}")
    return None

##Functions are put into a list for a loop:
args = [(f,df,3), (y,dy,1), (h,dh,3), (G, dG, 1)]

# Modified Testing code from assignment that allows looping over Functions
# Using a list of tuples of the function, its derivative, and varying initial guesses.
for F,dF,x0 in args:
    #Call the Newton function to approximate a root given the ordered parameters from args
    xn = Newton(F, dF, x0, 1e-10)
    if (type(xn) is float):
        #Using .__name__ method to get the name of the function currently in use
        print(f"{F.__name__}(%1.8f) = %1.8f using x0 = {x0}"%(xn,F(xn)))
    else:
        print(f"Newton’s method failed with initial point {x0}.")

#Problem 2 text description:
    # Using the provided function h(x) = x^5-5x^3+4x+1 and the derivative,
    #   5x^4 -15x^2 + 4, a selected starting point can converge using newtons method
    #   and an approxomated root can be returned. At x_0 = {1, 2, 3}, the root converges at
    #   1.95407961.. (approxomated). At x_0 = -5, the approxomated root was -2.03849529, and for
    #    x_0 = 0 yielded an approxomated root of -.27583419
    #   for an x_0 =1000, the function exceeded trial limit.
