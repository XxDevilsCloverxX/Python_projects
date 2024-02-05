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

    # Compute the within-class variance
    within_variance = grouped_vars.multiply(P_js, axis='index').sum(axis='columns') # Sum up each row after multiplying the probability with variance
    
    # describe between class variance
    between_variance = ((grouped_means.subtract(description.loc['mean']))**2).multiply(P_js, axis='index').sum(axis='columns')
    return description, within_variance, between_variance

def batchPerceptron(epochs:int, dataset:np.ndarray, seed:int):
    """
    @purpose:
        Perform the batch perceptron over a dataset to compute weights
        All pre-processing (Class 2 x -1, etc) needs to occur before this call
    @param:
        itterations - maximum itterations the perceptron should run for
        dataset - dataset that will generate the weights
    @return:
        weights - final weights computed from perceptron
        k - # of itterations needed to converge
        len(misclassifications) - # of misclassified features
    """
    np.random.seed(seed) # set the seed
    # take an initial guess of the weights based on the columns provided
    weights = np.random.normal(loc=0,scale=1, size=(dataset.shape[1],1))
    rho_k = 1e-1 # set an initial learning rate
    
    for k in range(1,epochs+1):
        missclassifications = []
        rho_k *= .8
        for row in dataset:        
            # compute the class of the row with the weight vector
            classification = weights.T.dot(row)
            # if the classification is negative, append to the missclassifications
            if classification <= 0:
                missclassifications.append(row) # append the data point that was misclassed
        
        # if there were no misclassifications in the entire dataset, complete
        if len(missclassifications) == 0:
            print(f'Batch Perceptron converged in {k} epochs')
            return k, len(missclassifications), weights
        # If the previous condition fails, we need to sum all of the misclassifications then adjust weight
        sum_missclassifications = np.sum(missclassifications, axis=0).reshape(-1, 1)
        weights += rho_k * sum_missclassifications
    print(f'Batch Perceptron failed to converge in {epochs} epochs')
    
    return k, len(missclassifications), weights

def leastSquaresClassifier(labels:np.ndarray, data:np.ndarray):
    """
    @purpose:
        compute the weights based on least squares
    @param:
        labels - target vector mapping data to classes
        data - input matrix of values appended or prepended with ones
    @return:
        weights- weights that map data to labels
    """
    weights = np.linalg.pinv(data).dot(labels)
    return weights

def ComputeMisclassLS(weights:np.ndarray, data:np.ndarray):
    """
    @purpose:
    @param:
        weights- weights computed from LS
        data - X matrix of data
    @return:
        number of points where data.dot(weights) returns 0 or negative values
    """
    predictions = data.dot(weights)
    # Convert predictions to binary (0 if positive, 1 if negative)
    binary_predictions = np.where(predictions >= 0, 0, 1)
    # Count misclassifications
    num_misclass = np.sum(binary_predictions)
    return num_misclass    

def plotStatistics(df:pd.DataFrame):
    """
    @purpose:
        Take an input data frame and plot some characteristics of its data
        helper function to clean up the main driver function
    """
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
    return df

def main(excel:str, limit:int):
    """
    @purpose:
        Driver function for control flow of the program
    """
    df = pd.read_excel(excel)
    df.rename(columns={'meas_1': 'SepalL', 'meas_2': 'SepalW', 'meas_3': 'PetalL', 'meas_4': 'PetalW'}, inplace=True)
    df_description, within_class_variance, between_class_variance = describeData(df)
    print(f'Data Statistics:\n{df_description}\n')
    print(f'Within-Class Variance:\n{within_class_variance}\n')
    print(f'Between-Class Variance:\n{between_class_variance}\n')
    
    # Call the plotting helper function
    df = plotStatistics(df=df)

    # Standardize the data - preprocessing for batch perceptron
    features = df.drop(columns='species')
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(features)
    df_standardized = pd.DataFrame(standardized_data, columns=features.columns)
    df_standardized['species'] = df['species']  # Add the species column back

    # Performing the Setosa vs Others, All features, Batch Perceptron and LS classifications
    original_data = df.copy()
    labels = original_data['species'].to_numpy().reshape(-1,1)

    setosa_vs_others = df_standardized.copy()
    mask = df_standardized['species'] != 1
    setosa_vs_others.loc[mask, setosa_vs_others.columns != 'species'] *= -1
    setosa_vs_others = setosa_vs_others.drop(columns='species').to_numpy()
    setosa_vs_others = np.c_[np.ones(setosa_vs_others.shape[0]), setosa_vs_others]  # prepend a col of ones 
    k, num_misclass, weights_BP = batchPerceptron(epochs=limit, dataset=setosa_vs_others, seed=0)
    weights_LS = leastSquaresClassifier(labels=labels, data=setosa_vs_others)
    misclassed_LS = ComputeMisclassLS(weights=weights_LS, data=setosa_vs_others)
    print(f'{num_misclass} misclassed datapoints BP, \nWeights BP: \n{weights_BP}')
    print(f'{misclassed_LS} misclassed datapoints LS, \nWeights LS: \n{weights_LS}')
    
    

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        prog='Rodriguez_Silas_Project1.py',
        description='Linear Classifiers on fisheriris dataset',
        epilog='Dep: Numpy, seaborn, matplotlib, pandas')

    # Add the CSV + max itterations arguments
    parser.add_argument('-f','--excel',
                         type=str,
                         help='Relative path to excel file',
                         default='Proj1DataSet.xlsx')
    
    parser.add_argument('-l', '--limit',
                        type=int,
                        metavar='MAX',
                        help='Maximum itterations to use for gradient descent (default 1000)',
                        default=1000)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided Excel file
    main(args.excel, args.limit)