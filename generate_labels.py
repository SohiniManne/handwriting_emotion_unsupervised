import numpy as np
from sklearn.cluster import KMeans

features = np.load("features.npy")

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features)

# ✅ SAVE clusters for visualization
np.save("clusters.npy", clusters)

# Automatic emotion mapping
cluster_centers = kmeans.cluster_centers_
order = np.argsort(cluster_centers[:, 0])

emotion_map = {
    order[0]: 1,  # Sad
    order[1]: 3,  # Neutral
    order[2]: 0,  # Happy
    order[3]: 2   # Angry
}

labels = np.array([emotion_map[c] for c in clusters])
np.save("pseudo_labels.npy", labels)

print("✔ clusters.npy and pseudo_labels.npy generated")
