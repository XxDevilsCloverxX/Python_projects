import numpy as np
import pandas as pd

####################
"""
Functions for the Bayes CLF
"""
def multivariate_gaussian(x, mean, cov):
    """
    Compute the multivariate Gaussian PDF for a given feature vector x.
    
    Parameters:
        x: Feature vector.
        mean: Mean vector of the Gaussian distribution.
        cov: Covariance matrix of the Gaussian distribution.
        
    Returns:
        pdf: Probability density function value.
    """
    d = len(x)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    coef = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    pdf = coef * np.exp(exponent)
    return pdf

def BayesCLF(testfile, m, s, p):
    # open the test dataset
    df_test = pd.read_excel(testfile, header=None, names=('x1', 'x2', 'x3', 'x4', 'x5', 'label'))
    df_testy = df_test.iloc[:, -1].to_numpy()
    df_testx = df_test.iloc[:, :-1].to_numpy()
    
    # take the prior
    priors = p
    # take the given model means for each class
    means = m
    # take the given covar matrices for each class
    covs = s
    # store possible classes
    classes = df_test['label'].unique()

    # initialize array to store posterior probabilities
    posterior_probs = np.zeros((len(df_testx), len(classes)))
    
    # calculate posterior probabilities for each test sample and each class
    for i, x in enumerate(df_testx):
        for j, _ in enumerate(classes):
            prior = priors[j]
            mean = means[j]
            cov = covs[j]
            likelihood = multivariate_gaussian(x, mean, cov)
            posterior_probs[i, j] = prior * likelihood
    
    # normalize posterior probabilities
    posterior_probs /= np.sum(posterior_probs, axis=1, keepdims=True)

    # make predictions based on maximum posterior probability
    predicted_labels = 1 + np.argmax(posterior_probs, axis=1)
    error = np.mean(predicted_labels != df_testy)
    return error
####################
####################
"""
Functions for Naive Bayes CLF
"""
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
####################
####################
"""
Functions for the MLE Bayes CLF
"""
def MLE_estimation(trainfile, testfile):
    # open the datasets
    df = pd.read_excel(trainfile, header=None ,names=('x1', 'x2', 'x3', 'x4', 'x5', 'label'))
    
    # estimate the prior
    priors = (df.groupby('label').size() / len(df)).to_numpy()
    # get the parameters of the model
    classes = df['label'].unique()
    num_classes = len(classes)
    num_features = df.shape[1] - 1
    
    # Initialize arrays for means and covariance matrices
    means = np.zeros((num_classes, num_features))
    covs = np.zeros((num_classes, num_features, num_features))
    
    # Estimate means and covariance matrices for each class
    for i, cls in enumerate(classes):
        class_data = df[df['label'] == cls].iloc[:, :-1]
        class_mean = class_data.mean().values
        class_cov = np.cov(class_data.T)
        means[i] = class_mean
        covs[i] = class_cov

    return BayesCLF(testfile, means, covs, priors)
####################

def main():

    # Initialize the distribution params
    m = np.vstack((np.zeros(5), np.ones(5)))  # Mean vectors for class 1 and class 2
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
    p = np.array((1/2, 1/2))  # Class probabilities

    # Create an empty DataFrame
    errors = pd.DataFrame(index=['Naive Bayes', 'MLE Bayes', 'Bayes'], 
                      columns=['Error Train_N = 100', 'Error Train_N = 1000'])

    errors.at['Naive Bayes', 'Error Train_N = 100']  = NaiveBayesCLF('Proj4Train100.xlsx', 'Proj4Test.xlsx')
    errors.at['Naive Bayes', 'Error Train_N = 1000'] = NaiveBayesCLF('Proj4Train1000.xlsx', 'Proj4Test.xlsx')
    errors.at['MLE Bayes', 'Error Train_N = 100']  = MLE_estimation('Proj4Train100.xlsx', 'Proj4Test.xlsx')
    errors.at['MLE Bayes', 'Error Train_N = 1000'] = MLE_estimation('Proj4Train1000.xlsx', 'Proj4Test.xlsx')
    errors.at['Bayes', 'Error Train_N = 100']  = BayesCLF('Proj4Test.xlsx', m,s,p)
    errors.at['Bayes', 'Error Train_N = 1000'] = BayesCLF('Proj4Test.xlsx', m,s,p)
    print(errors)

if __name__ == '__main__':
    main()