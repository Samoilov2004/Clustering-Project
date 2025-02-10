import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_samples = 500
n_features = 2

X_3, y_3 = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, cluster_std=1.5, random_state=42)
X_4, y_4 = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, cluster_std=1.5, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ssd = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_3)
    ssd.append(kmeans.inertia_)

axes[0].plot(K, ssd, 'bo-', markersize=8, linewidth=2, alpha=0.7)
axes[0].set_xlabel('Число кластеров', fontsize=14)
axes[0].set_xticks(K)
axes[0].grid(True)

optimal_k = 3
axes[0].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, alpha=0.7)
axes[0].text(optimal_k + 0.1, max(ssd) * 0.9, f'Оптимальное k = {optimal_k}', color='r', fontsize=12);

ssd = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_4)
    ssd.append(kmeans.inertia_)

axes[1].plot(K, ssd, 'bo-', markersize=8, linewidth=2, alpha=0.7)
axes[1].set_xlabel('Число кластеров', fontsize=14)
axes[1].set_xticks(K)
axes[1].grid(True)

optimal_k = 4
axes[1].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, alpha=0.7)
axes[1].text(optimal_k + 0.1, max(ssd) * 0.9, f'Оптимальное k = {optimal_k}', color='r', fontsize=12);

plt.tight_layout()
plt.savefig('file2.png')
plt.show()