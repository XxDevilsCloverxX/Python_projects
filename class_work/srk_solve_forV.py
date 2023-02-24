R= .0821
w = .225
Tc = 304.2
Pc = 72.9
m = .48508 + 1.5517*w-.1561*w**2
a = .42747 * ((R*Tc)**2)/Pc
b = .08664 * (R*Tc)/Pc

print("a:", a, "\nb:", b)
def ideal_law(temp, pressure):
    v_ideal = R * temp / pressure
    return v_ideal
def find_vhat(temp, pressure, alpha):
    v_hat = .05 # Guess from v_ideals
    upp_bnd = .5
    low_bnd = -.5
    condition = True
    while (condition ==True):
        x = pressure - ((R*temp)/(v_hat-b)) + ((alpha*a)/v_hat*(v_hat-b))
        if (x>= low_bnd and x<=upp_bnd):
            return v_hat
            condition = False
        elif (x > upp_bnd):
            v_hat = v_hat - .04
        elif (x < low_bnd):
            v_hat += .1
        else:
            v_hat+=.02

#SRK Equation solving
import pandas as pd
temp_array=[200,250,300,300,300]
pressure_array=[6.8,12.3,6.8,21.5,50]
arrays_len = len(temp_array)
Tr_arr = []
v_ideal = []
alpha_values = []

#For ideal laws values & Tr values
for i in range(0, arrays_len):
    x = ideal_law(temp_array[i], pressure_array[i])
    v_ideal.append(x)
    y = temp_array[i] / Tc
    Tr_arr.append(y)

#For alpha values per equation
for j in range(0, arrays_len):
    temp_value = Tr_arr[j]
    alpha = (1+m*(1-(temp_value**.5)))**2
    alpha_values.append(alpha)
v_srk_arr = []
for k in range(0, arrays_len):
    V_srk = find_vhat(temp_array[k], pressure_array[k], alpha_values[k])
    v_srk_arr.append(V_srk)
difference = []
for l in range(0, arrays_len):
    percent = round(((v_ideal[l]-v_srk_arr[l])/v_srk_arr[l]) *100, 2)
    percent = str(percent)
    percent += "%"
    difference.append(percent)
data = {
    "T(K)": temp_array,
    "P(atm)": pressure_array,
    "alpha": alpha_values,
    "V(Ideal)": v_ideal,
    "V(SRK)": v_srk_arr,
    "Difference": difference
}
dataframe = pd.DataFrame(data)
dataframe.index = dataframe.index +1
print(dataframe)
