import numpy as np
import matplotlib.pyplot as plt

features = np.load("features.npy")

labels = ["Mean Intensity (Pressure)",
          "Edge Density (Stroke Complexity)",
          "Texture Roughness"]

plt.figure(figsize=(12,4))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.hist(features[:,i], bins=40)
    plt.title(labels[i])
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
