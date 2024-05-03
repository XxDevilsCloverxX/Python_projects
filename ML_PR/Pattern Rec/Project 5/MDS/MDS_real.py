import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

#bigger or smaller dataset
small = False

# Load the NxN distance dataset - training data
df = pd.read_excel('texas_cities.xlsx', index_col=0) if small else pd.read_excel('more_texas_cities.xlsx', index_col=0)

# Pop 'Ft_Worth' row and column
ft_worth_row = df.loc['Ft. Worth']
ft_worth_col = df['Ft. Worth']
df = df.drop('Ft. Worth', axis=0)  # Drop row
df = df.drop('Ft. Worth', axis=1)  # Drop column

# Verify the shape of the data
print(df.head())
print(df.shape)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df)

# Perform MDS
mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1, random_state=69)
mds_result = mds.fit_transform(df)

# Find the index of 'Austin'
austin_index = df.index.get_loc('Austin')

# Plotting
plt.figure(figsize=(12, 6))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue')
for i, city in enumerate(df.index):
    plt.annotate(city, (pca_result[i, 0], pca_result[i, 1]))
plt.title('PCA Plot')

# MDS Plot
mds_result
plt.subplot(1, 2, 2)
plt.scatter(mds_result.T[0, :], mds_result.T[1, :], c='red')
plt.scatter(mds_result.T[0, austin_index], mds_result.T[1, austin_index], marker='*', color='blue', label='Austin')
for i, city in enumerate(df.index):
    plt.annotate(city, (mds_result.T[0, i], mds_result.T[1, i]))
plt.title('MDS Plot')
plt.legend()

plt.tight_layout()
plt.show()

# Create an empty row of NaNs with the correct dtype
empty_row = pd.Series(index=df.columns, dtype=np.float64)

# Add 'Ft_Worth' row and column to the DataFrame
df.loc['Ft_Worth'] = np.nan
df['Ft_Worth'] = np.nan

# Replace NaNs with 'Ft_Worth' row and column
df.loc['Ft_Worth'] = ft_worth_row.values
df['Ft_Worth'] = ft_worth_col.values

print(df.tail())
print(df.shape)
# Fit_transform MDS
mds_result_with_ft_worth = mds.fit_transform(df)

# Plot MDS with Ft_Worth flipped 180 degrees
plt.figure(figsize=(10, 6))
plt.scatter(-mds_result_with_ft_worth[:, 0], -mds_result_with_ft_worth[:, 1], c='red')  # Flipping coordinates
plt.scatter(-mds_result[austin_index, 0], -mds_result[austin_index, 1], marker='x', color='green', label='Old Apx. Austin')  # Flipping coordinates
plt.scatter(-mds_result_with_ft_worth[austin_index, 0], -mds_result_with_ft_worth[austin_index, 1], marker='*', color='blue', label='New Apx. Austin')  # Flipping coordinates
for i, city in enumerate(df.index):
    plt.annotate(city, (-mds_result_with_ft_worth[i, 0], -mds_result_with_ft_worth[i, 1]))  # Flipping coordinates
plt.title('MDS Plot with Ft_Worth (Flipped 180 degrees)')
plt.legend()
plt.tight_layout()
plt.show()


# Show pairwise distances in original space
df = df.astype(int)
print("Pairwise distances in original space:")
print(f"{df=}", end="\n\n")

# Show pairwise distances in mapped space
mapped_distances_df = pd.DataFrame(pairwise_distances(mds_result_with_ft_worth))
mapped_distances_df = np.round(mapped_distances_df, decimals=1)
mapped_distances_df.index = df.index  # Map original indices
mapped_distances_df.columns = df.index  # Map original columns
print("Pairwise distances in mapped space:")
print(mapped_distances_df)