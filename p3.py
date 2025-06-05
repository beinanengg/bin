import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df["Target"] = y

plt.figure(figsize=(10, 7))
for i, label in enumerate(iris.target_names):
    subset = df[df["Target"] == i]
    plt.scatter(subset["PCA1"], subset["PCA2"], label=label, alpha=0.8)


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.grid()
plt.show()

explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Total Explained Variance: {np.sum(explained_variance):.2f}")