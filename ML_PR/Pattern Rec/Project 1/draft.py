from pandas import read_excel, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def LS_2_Classifier(X:np.ndarray, t:np.ndarray):
    """
    @purpose:
        compute the weights as a result of the least-squares classifier
    @pre:
        X must be augmented with a column of ones
        t must be selected as -1 or 1
    @param:
        X - input dataframe
        t - labels
    @return:
        weights - vector of weights computed from the LS-solution
    """
    weights = np.linalg.pinv(X).dot(t)
    return weights

def BatchPerceptron(X:np.ndarray, epochs=1000):
    """
    @purpose:
        - try to compute the weights that would classify X correctly
    @pre:
        - X be a 2 class problem, with features in class 2 being scaled by -1
        - X is to be augmented with a column of 1s
    @return:
        weights- the weights that correctly all features of class X, if convergence
        k - epoch that BP converged on
    """
    weights = np.random.normal(loc=0, scale=1.0, size=X.shape[1]) # initial guess
    rho_k = 1 # initial learning rate
    for k in range(1,epochs+1):
        # compute the predictions
        predictions = X.dot(weights)
        # get the indeces of negative (misclassed) features
        bad_classes = predictions<=0
        # print(X[bad_classes].shape) # this should shrink over time
        # if no misclassifications, convergence
        if not np.any(predictions<=0):
            print(f'Converged at {k} epochs')
            return k, weights
        weights += rho_k * np.sum(X[bad_classes], axis=0)
        rho_k *= 0.8
    print(f'Batch Perceptron Failed to Converge in {epochs} epochs')
    return k, weights

def ComputeMisclass_LS_Multi(W:np.ndarray, X:np.ndarray, T:np.ndarray):
    """
    @purpose:
        Report the misclassifications of LS_Multi
    @pre:
        W - weight matrix (l+1) x M
        X - Data with bias N x (l+1)
        T - one hot labels N x M
    @return:
        count - misclassified points count
        accuracy - accuracy of the LS _ Multi classifier
    """
    prediction_matrix = X.dot(W)    # should be N x M predictions, where the Mth column with the max value in row N is the class of N
    prediction_one_hot = np.eye(W.shape[1])[np.argmax(prediction_matrix, axis=1)]
    # Find misclassified points
    misclassified_indices = np.where(~np.all(T == prediction_one_hot, axis=0))[0]
    # report the accuracy with the count
    count = len(misclassified_indices)
    accuracy = 1 - count / T.shape[0]
    return count, accuracy

def ComputeMisclass_BP(w:np.ndarray, X:np.ndarray, t:np.ndarray):
    """
    @purpose:
        compute the misclassifications of the BP regardless of convergence
    @pre:
        weights (l+1) x 1
        X be the biased features N x (l+1)
        t is the labels for classes : 1,2 - N x 1
    """
    prediction_matrix = X.dot(w)
    classes = np.where(prediction_matrix > 0, 1, 2)   # if above threshold, give me a class 2, else class 1
    misclasses = np.sum(classes!=t)
    accuracy = 1 - misclasses / t.shape[0]   # get the accuracy from correct classes
    return misclasses, accuracy

def ComputeMisclass_LS(w:np.ndarray, X:np.ndarray, t:np.ndarray):
    """
    @purpose:
        Compute the misclassifications and accuracy of decisions made from the LS method
    @pre:
        w - 1 x (l+1)
        X - N x (l+1)
        t - N x 1
    """
    # compute the prediction vector
    predictions = X.dot(w)
    predicted_labels = np.round(predictions, 0)
    misclasses = np.sum(predicted_labels!=t)
    accuracy = 1 - misclasses / t.shape[0]   # get the accuracy from correct classes
    return misclasses, accuracy

# Seed the program
np.random.seed(0)

# Open the dataset
df = read_excel('Proj1DataSet.xlsx').to_numpy()

# Extract features and labels
features = df[:, :-1]
labels_text = df[:, -1]

# Define class mapping
class_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

# Convert text labels to numerical values
labels = np.array([class_mapping[label] for label in labels_text])

# # Describe the data
# min_values = np.min(features, axis=0)
# max_values = np.max(features, axis=0)
# mean_values = np.mean(features, axis=0)
# variance_values = np.var(features, axis=0)

# # Print the results
# print("Minimum values:", min_values)
# print("Maximum values:", max_values)
# print("Mean values:", mean_values)
# print("Variance values:", variance_values)

# # Get each bin's frequency in the data
# P_js = np.bincount(labels) / labels.shape[0]
# clusters = np.unique(labels)

# # Compute the within-class variance
# Sw = []
# for k, cluster in enumerate(clusters):
#     # isolate the data of this cluster:
#     feature_cluster = features[labels==cluster]
#     # compute each column's variance
#     variance_values = np.var(feature_cluster, axis=0)
#     # multiply with the P_j of this corresponding class
#     variance_values *= P_js[k]
#     Sw.append(variance_values)
# Sw = np.sum(Sw, axis=0)
# print(f'Within-Class Variance: {Sw}')

# # Compute the between-class variance
# Sb = []
# for k, cluster in enumerate(clusters):
#     # isolate the data of this cluster:
#     feature_cluster = features[labels==cluster]
#     # compute each column's variance
#     variance_values = np.power(np.mean(feature_cluster, axis=0) - mean_values, 2)
#     # multiply with the P_j of this corresponding class
#     variance_values *= P_js[k]
#     Sb.append(variance_values)
# Sb = np.sum(Sb, axis=0)
# print(f'Between-Class Variance: {Sb}')

# # Compute the correlation matrix
# df[:, -1] = labels
# col_names = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Species']
# df = DataFrame(df,columns=col_names)
# corr_matrix = df.corr()
# plt.figure(figsize=(10,8))    # create a 10 in x 8 in figure
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
# plt.title('Correlation Heatmap')

# # Plot each feature against the class label in a loop
# features_columns = df.columns[:-1]  # Exclude the last column (label)
# num_features = len(features_columns)

# plt.figure(figsize=(10, 8))

# for i, feature in enumerate(features_columns, start=1):
#     plt.subplot(2, 2, i)
#     plt.scatter(df[feature], df['Species'], marker='x', color='red')
#     plt.title(f'{feature} vs Species')
#     plt.xlabel(feature)
#     plt.ylabel('Species')
#     plt.xticks(np.arange(0,9,step=2))

# plt.tight_layout()
# plt.show()
###################################################################################################
# Redefine class mapping
class_mapping = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
labels +=1

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Divide the standardized features based on their labels, augment a col of ones
setosa = features_standardized[labels == class_mapping['setosa']]
setosa = np.c_[np.ones(setosa.shape[0]), setosa]
versi = features_standardized[labels == class_mapping['versicolor']]
versi = np.c_[np.ones(versi.shape[0]), versi]
virgin = features_standardized[labels == class_mapping['virginica']]
virgin = np.c_[np.ones(virgin.shape[0]), virgin]
# Section Break
print()
###################################################################################################
# Compute the weights for LS method on setosa vs others All Features
# print('Computing Boundaries for Setosa vs others All Features...')
# other = np.vstack((versi, virgin))
# X_all = np.vstack((setosa, versi, virgin))
# L_all = np.where(labels != class_mapping['setosa'], 2, 1)

# weights_LS = LS_2_Classifier(X=X_all, t=L_all)
# misclasses, accuracy = ComputeMisclass_LS(w=weights_LS, X=X_all, t=L_all)
# print(f'{misclasses} misclassifications from Least-Squares with accuracy {accuracy*100}%.')

# # Preprocess X for batch perceptron
# X_batch = X_all[L_all != 1, :] * -1
# k, weights_BP = BatchPerceptron(X=X_batch)
# misclasses, accuracy = ComputeMisclass_BP(w=weights_BP, X=X_all, t=L_all)
# print(f'{misclasses} misclassifications from Batch-Perceptron with accuracy {accuracy*100}%.')

# print(weights_BP.shape, weights_BP)
# print(weights_LS.shape, weights_LS)

# # Section Break
# print()
# ##################################################################################################3
# Compute the weights for LS method on setosa vs others features 3 & 4 only
print('Computing Boundaries for Setosa vs others Features 3 & 4 only...')
X_f34 = np.vstack((setosa[:, [0, 3, 4]], versi[:, [0, 3, 4]], virgin[:, [0, 3, 4]]))
L_f34 = np.where(labels != class_mapping['setosa'], 2, 1)

weights_LS = LS_2_Classifier(X=X_f34, t=L_f34)
misclasses, accuracy = ComputeMisclass_LS(w=weights_LS, X=X_f34, t=L_f34)
print(f'{misclasses} misclassifications from Least-Squares with accuracy {accuracy*100}%.')

# Preprocess X for batch perceptron
# Preprocess X for batch perceptron
X_class1 = X_f34[L_f34 == 1, :]
X_class2_scaled = X_f34[L_f34 != 1, :] * -1
X_batch = np.vstack((X_class1, X_class2_scaled))
print(X_batch.shape)
k, weights_BP = BatchPerceptron(X=X_batch)
misclasses, accuracy = ComputeMisclass_BP(w=weights_BP, X=X_f34, t=L_f34)
print(f'{misclasses} misclassifications from Batch-Perceptron with accuracy {accuracy*100}%.')


# plot least squares vs scatter data
xline = np.linspace(X_f34[:, 1].min(), X_f34[:, 1].max())
yline_LS = -(weights_LS[0] + weights_LS[1] * xline) / weights_LS[2]
yline_BP = -(weights_BP[0] + weights_BP[1] * xline) / weights_BP[2]

print(weights_BP.shape, weights_BP)
print(weights_LS.shape, weights_LS)

plt.scatter(X_f34[:, 1], X_f34[:, 2], c=L_f34, cmap='bwr', marker='o')
plt.plot(xline, yline_LS, color='green', label='Least-Squares d(x)')
plt.plot(xline, yline_BP, color='black', label=f'Batch-Perceptron d(x) - {k}')
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.colorbar(label='Class Labels')
plt.legend()
plt.show()

# # Section Break
# print()
##################################################################################################
# # Compute the weights for LS method on virgi vs others All features
# print('Computing Boundaries for virginica vs others All Features...')
# X_all = np.vstack((setosa, versi, virgin))
# L_all = np.where(labels != class_mapping['virginica'], 2, labels)
# L_all[labels==class_mapping['virginica']] = 1    # set virgi to class 1, others to class 2

# weights_LS = LS_2_Classifier(X=X_all, t=L_all)
# misclasses, accuracy = ComputeMisclass_LS(w = weights_LS, X=X_all, t = L_all)
# print(f'{misclasses} misclassifications from Least-Squares with accuracy {accuracy*100}%.')

# print(X_all.shape, L_all.shape)

# # Preprocess X for batch perceptron
# print(np.c_[X_all, L_all])
# X_batch = X_all[L_all != 1, :] * -1
# print(np.c_[X_all, L_all])
# k, weights_BP = BatchPerceptron(X=X_batch)
# misclasses, accuracy = ComputeMisclass_BP(w = weights_BP, X=X_all, t = L_all)
# print(f'{misclasses} misclassifications from BP with accuracy {accuracy*100}%.')

# print(weights_BP.shape, weights_BP)
# print(weights_LS.shape, weights_LS)

# # Section Break
# print()

# # Compute the weights for LS method on virgi vs others features 3 & 4
# print('Computing Boundaries for virginica vs others Features 3 & 4 only...')
# other = np.vstack((setosa, versi))
# X_f34 = np.vstack((setosa[:, [0,3,4]], versi[:, [0,3,4]], virgin[:, [0,3,4]]))
# F34_L = np.where(labels != class_mapping['virginica'], 2, labels)
# F34_L[labels==class_mapping['virginica']] = 1    # set virgi to class 1, others to class 2

# weights_LS = LS_2_Classifier(X=X_f34, t=F34_L)
# misclasses, accuracy = ComputeMisclass_LS(w = weights_LS, X=X_f34, t = F34_L)
# print(f'{misclasses} misclassifications from Least-Squares with accuracy {accuracy*100}%.')

# # Preprocess X for batch perceptron
# X_f34[F34_L!=1, :] *= -1
# k, weights_BP = BatchPerceptron(X=X_f34)
# # Reprocess X for error computation
# X_f34[F34_L!=1, :] *= -1
# misclasses, accuracy = ComputeMisclass_BP(w = weights_BP, X=X_f34, t = F34_L)
# print(f'{misclasses} misclassifications from BP with accuracy {accuracy*100}%.')

# print(weights_BP.shape, weights_BP)
# print(weights_LS.shape, weights_LS)

# # plot least squares vs scatter data
# xline = np.linspace(X_f34[:,1].min(), X_f34[:,1].max())
# yline_LS = -(weights_LS[0] + weights_LS[1] * xline) / weights_LS[2]
# yline_BP = -(weights_BP[0] + weights_BP[1] * xline) / weights_BP[2]

# plt.scatter(virgin[:, 3], virgin[:, 4], marker='x', color='red')
# plt.scatter(other[:, 3], other[:, 4], marker='^', color='green')
# plt.plot(xline, yline_LS, color='black', label='Least-Squares d(x)')
# plt.plot(xline, yline_BP, color='blue', label=f'Batch-Perceptron d(x) - {k}')
# plt.xlabel('Feature 3')
# plt.ylabel('Feature 4')
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.legend()
# plt.show()

# # Section Break
# print()

# # Multiclass LS on features 3 & 4 only
# print('Computing Boundaries for Setosa vs Versi vs Virgi Features 3 & 4 only...')
# X_f34 = np.vstack((setosa[:, [0,3,4]], versi[:, [0,3,4]], virgin[:, [0,3,4]]))  # collect our X matrix
# unique_labels = len(np.unique(labels))
# T_matrix = np.eye(unique_labels)[labels-1]

# # Each w vect within W corresponds to one class
# W_Matrix = np.linalg.pinv(X_f34).dot(T_matrix)
# # print(W_Matrix.shape, W_Matrix) # verify this guy is (l+1) x M : (3 x 3)

# misclasses, accuracy = ComputeMisclass_LS_Multi(W = W_Matrix, X=X_f34, T=T_matrix)
# print(f'{misclasses} misclassifications with accuracy {accuracy*100}%.')

# print(W_Matrix.shape, W_Matrix)

# # Prepare plotting
# xline = np.linspace(X_f34[:, 1].min(), X_f34[:, 1].max())
# y1_line = -(W_Matrix[0, 0] + W_Matrix[1, 0] * xline) / W_Matrix[2, 0]
# y2_line = -(W_Matrix[0, 1] + W_Matrix[1, 1] * xline) / W_Matrix[2, 1]
# y3_line = -(W_Matrix[0, 2] + W_Matrix[1, 2] * xline) / W_Matrix[2, 2]

# d12 = y1_line - y2_line
# d13 = y1_line - y3_line
# d23 = y2_line - y3_line

# plt.scatter(setosa[:, 3], setosa[:, 4], marker='^', color='black', label=f'Class {class_mapping["setosa"]}')
# plt.scatter(versi[:, 3], versi[:, 4], marker='o', color='blue', label=f'Class {class_mapping["versicolor"]}')
# plt.scatter(virgin[:, 3], virgin[:, 4], marker='x', color='red', label=f'Class {class_mapping["virginica"]}')

# plt.plot(xline, d12, color='green', label='d12')
# plt.plot(xline, d13, color='purple', label='d13')
# plt.plot(xline, d23, color='orange', label='d23')
# plt.xlabel('Feature 3')
# plt.ylabel('Feature 4')
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.legend()
# plt.show()