import numpy as np
from sklearn.preprocessing import StandardScaler

# Create a 2D matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Standardize along columns
scaler = StandardScaler()
matrix_standardized = scaler.fit_transform(matrix)

# Print the original and standardized matrices
print("Original Matrix:")
print(matrix)
print("\nStandardized Matrix:")
print(matrix_standardized)

print("Mean values for Setosa:", np.mean(matrix_standardized , axis=0))
print("std values for Versicolor:", np.std(matrix_standardized, axis=0))