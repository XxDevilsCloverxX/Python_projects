import numpy as np
import pandas as pd

def gaussian(X, mu, var):
    # Calculate the exponent term
    exponent = -((X[:, np.newaxis] - mu) ** 2) / (2 * var)
    
    # Calculate the Gaussian probability density function
    pdf = np.exp(exponent) / np.sqrt(2 * np.pi * var)
    
    # Compute the product of probabilities along each feature axis
    return np.prod(pdf, axis=2)

def NaiveBayesCLF(trainfile, testfile):
    # open the datasets
    df = pd.read_excel(trainfile, header=None ,names=('x1', 'x2', 'x3', 'x4', 'x5', 'label'))
    df_test = pd.read_excel(testfile, header=None)
    df_testy = df_test.iloc[:, -1].to_numpy()
    df_testx = df_test.iloc[:, :-1].to_numpy()
    
    # estimate the prior
    priors = (df.groupby('label').size() / len(df)).to_numpy()
    # print(priors)
    # estimate the means
    means = df.groupby('label').mean().to_numpy()
    # estimate the variances
    vars = df.groupby('label').var().to_numpy()
    # print(vars)
    # store possible classes
    classes = df['label'].unique()
    # print(classes)
    
    # get the test scores
    tst_z = gaussian(df_testx, means, vars)
    # multiply by priors
    probs = tst_z * priors
    # class predictions
    pred_y = 1 + np.argmax(probs, axis=1)
    
    # compute the error
    error = np.mean(pred_y != df_testy)
    return error

def main():

    # Create an empty DataFrame
    errors = pd.DataFrame(index=['Naive Bayes', 'MLE Bayes', 'Bayes'], 
                      columns=['Error Train_N = 100', 'Error Train_N = 1000'])

    errors.at['Naive Bayes', 'Error Train_N = 100'] = NaiveBayesCLF('Proj4Train100.xlsx', 'Proj4Test.xlsx')
    errors.at['Naive Bayes', 'Error Train_N = 1000'] = NaiveBayesCLF('Proj4Train1000.xlsx', 'Proj4Test.xlsx')
    print(errors)

if __name__ == '__main__':
    main()