import numpy as np
import pandas as pd
from NaiveBayes import NaiveBayesClassifier


def naive_bayes_clf(train_filename:str, test_filename:str):
    """
    train_filename:  str to the excel sheet to be read
    """
    df_test = pd.read_excel(test_filename, header=None)
    df = pd.read_excel(train_filename, header=None)
    df_grouped = df.groupby(df.columns[-1])

    # Convert DataFrame groups to NumPy arrays
    grouped_np = [data.to_numpy() for _, data in df_grouped]

    grouped_c1 = np.array(grouped_np[0])
    grouped_c2 = np.array(grouped_np[1])

    priors = np.array((grouped_c1.shape[0] / df.shape[0] , grouped_c2.shape[0] / df.shape[0]))
    # print(priors, priors.shape)


def main():
    # Define the parameters for the datasets
    m = np.vstack((np.zeros(5), np.ones(5)))  # Mean vectors for class 1 and class 2
    # print(m, m.shape)
    s = np.zeros((2,5,5))  # Covariance matrices for class 1 and class 2
    s[0,:,:] = np.array([[0.8, 0.2, 0.1, 0.05, 0.01],   # Covariance matrix for class 1
                        [0.2, 0.7, 0.1, 0.03, 0.02],
                        [0.1, 0.1, 0.8, 0.02, 0.01],
                        [0.05, 0.03, 0.02, 0.9, 0.01],
                        [0.01, 0.02, 0.01, 0.01, 0.8]])
    s[1,:,:] = np.array([[0.9, 0.1, 0.05, 0.02, 0.01],   # Covariance matrix for class 2
                        [0.1, 0.8, 0.1, 0.02, 0.02],
                        [0.05, 0.1, 0.7, 0.02, 0.01],
                        [0.02, 0.02, 0.02, 0.6, 0.02],
                        [0.01, 0.02, 0.01, 0.02, 0.7]])    
    # print(s, s.shape)
    p = np.vstack((1/2, 1/2))  # Class probabilities
    # print(p, p.shape)

    # perform classifiers for the given filename
    naive_bayes_clf('Proj4Train100.xlsx', 'Proj4Test.xlsx')
    # bayes_clfs('', m,s,p)
    pass

if __name__ == "__main__":
    main()