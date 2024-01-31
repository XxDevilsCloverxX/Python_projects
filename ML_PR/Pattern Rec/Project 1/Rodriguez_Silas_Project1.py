"""
    a) There are 3 classes to this problem: setosa, versicolor, and virginica
    b) There are 4 features: meas_1,meas_2,meas_3,meas_4: 
        Assuming this is sepal length, sepal width, pedal length, and pedal width
    c) After googling images and labels of what an iris sepal and pedal were, as well as
        examples on all three classes, I think there can be some classifactions that will perform
        okay, but I feel like there will be significant overlap with this classifier.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def describeData(dataset: pd.DataFrame):
    """
    @purpose: 
        Describe characteristics of the features such as mins, maxes,
        mean, variance, coorelation coefficients, 
    @param:
        dataset - pd.Dataframe for describing and obtaining these values
    @return:
        description - pd.Dataframe with the mins, maxes, means, and variance
    """
    # describe the features
    description = dataset.describe()
    variance = (description.loc['std'])**2
    description.loc['variance'] = variance
    # describe within class variance

    # describe between class variance
    between_var = (dataset.groupby('species').mean() - description.loc['mean'])**2
    between_var = between_var.sum().sum()   # sum all nums along rows and cols
    print(between_var)
    return description

def main(csv:str):
    """
    @purpose:
        Driver function for control flow of the program
    """
    df = pd.read_csv(csv)
    df_description = describeData(df)
    print(df_description)

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        prog='Rodriguez_Silas_Project1.py',
        description='Linear Classifiers on fisheriris dataset',
        epilog='Dep: Numpy, matplotlib, pandas, sklearn')

    # Add the CSV + max itterations arguments
    parser.add_argument('-f','--csv',
                         type=str,
                         help='Relative path to CSV file',
                         required=True)
    
    # parser.add_argument('-l', '--limit',
    #                     type=int,
    #                     metavar='MAX',
    #                     help='Maximum itterations to use for gradient descent (default 1000)',
    #                     default=1000)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided CSV file
    main(args.csv)