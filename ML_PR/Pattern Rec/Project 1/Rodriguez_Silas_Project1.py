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
import seaborn as sns
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
    # describe the features & compute variance
    description = dataset.describe()
    variance = (description.loc['std'])**2
    description.loc['variance'] = variance

    # Compute P_j terms for probability of a class : C data points / total data points
    frequencies = dataset['species'].value_counts()
    total = dataset['species'].count()
    P_js = frequencies / total # this is the P_j terms

    # Group the data by classes, then extract means and variances
    grouped_means = dataset.groupby('species').mean()
    grouped_vars = (dataset.groupby('species').std())**2
    print(grouped_means)
    # Compute the within-class variance
    within_variance = grouped_vars.multiply(P_js, axis='index').sum(axis='columns') # Sum up each row after multiplying the probability with variance
    
    # describe between class variance
    between_variance = ((grouped_means.subtract(description.loc['mean']))**2).multiply(P_js, axis='index').sum(axis='columns')
    return description, within_variance, between_variance

def main(excel:str):
    """
    @purpose:
        Driver function for control flow of the program
    """
    df = pd.read_excel(excel)
    df.rename(columns={'meas_1': 'SepalL', 'meas_2': 'SepalW', 'meas_3': 'PetalL', 'meas_4': 'PetalW'}, inplace=True)
    df_description, within_class_variance, between_class_variance = describeData(df)
    print(f'Data Statistics:\n{df_description}')
    print(f'Within-Class Variance:\n{within_class_variance}')
    print(f'Between-Class Variance:\n{between_class_variance}')

    # Create a mapping for each class to a numeric label
    class_mapping = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
    
    # Add a new column 'class_label' with the numeric labels
    df['species'] = df['species'].map(class_mapping)
    
    # compute & plot correlation coefficients for all the features with each other and plot to a heat map
    correlation_matrix = df.corr()
    plt.figure(figsize=(10,8))    # create a 10 in x 8 in figure
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap')

    # Plot scatter plots of each feature vs class mapping
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)    # create a 2x2 subplot top left
    plt.scatter(df['SepalL'], df['species'], marker='x', color='red')
    plt.title('SepalL vs Class')
    plt.xlabel('SepalL')
    plt.ylabel('Class')
    plt.xticks(np.arange(0,9,step=2))   # scale the graphs visually
    plt.yticks(np.arange(1,len(class_mapping)+.5,step=.5))

    plt.subplot(2,2,2)    # create a 2x2 subplot top right
    plt.scatter(df['SepalW'], df['species'], marker='x', color='red')
    plt.title('SepalW vs Class')
    plt.xlabel('SepalW')
    plt.ylabel('Class')
    plt.xticks(np.arange(0,9,step=2))   # scale the graphs visually
    plt.yticks(np.arange(1,len(class_mapping)+.5,step=.5))

    plt.subplot(2,2, 3)    # create a 2x2 subplot bottom left
    plt.scatter(df['PetalL'], df['species'], marker='x', color='red')
    plt.title('PetalL vs Class')
    plt.xlabel('PetalL')
    plt.ylabel('Class')
    plt.xticks(np.arange(0,9,step=2))   # scale the graphs visually
    plt.yticks(np.arange(1,len(class_mapping)+.5,step=.5))

    plt.subplot(2,2, 4)    # create a 2x2 subplot bottom left
    plt.scatter(df['PetalW'], df['species'], marker='x', color='red')
    plt.title('PetalW vs Class')
    plt.xlabel('PetalW')
    plt.ylabel('Class')
    plt.xticks(np.arange(0,9,step=2))   # scale the graphs visually
    plt.yticks(np.arange(1,len(class_mapping)+.5,step=.5))

    plt.tight_layout()  # adjust for better spacing
    plt.show()  # show both plots

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        prog='Rodriguez_Silas_Project1.py',
        description='Linear Classifiers on fisheriris dataset',
        epilog='Dep: Numpy, matplotlib, pandas, sklearn')

    # Add the CSV + max itterations arguments
    parser.add_argument('-f','--excel',
                         type=str,
                         help='Relative path to excel file',
                         default='Proj1DataSet.xlsx')
    
    # parser.add_argument('-l', '--limit',
    #                     type=int,
    #                     metavar='MAX',
    #                     help='Maximum itterations to use for gradient descent (default 1000)',
    #                     default=1000)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided Excel file
    main(args.excel)