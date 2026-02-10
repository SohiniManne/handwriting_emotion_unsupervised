import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

features = np.load("features.npy")
clusters = np.load("clusters.npy")  # from KMeans

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Handwriting Emotion Pattern Clusters (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster ID")
plt.show()
