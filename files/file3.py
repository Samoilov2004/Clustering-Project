import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_samples = 500
n_features = 2

X_3, y_3 = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, cluster_std=1.2, random_state=42)
X_4, y_4 = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, cluster_std=1.2, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_3)
    score = silhouette_score(X_3, kmeans.labels_)
    silhouette_scores.append(score)

axes[0].plot(K, silhouette_scores, 'bo-', markersize=8, linewidth=2, alpha=0.7)
axes[0].set_ylabel('Среднее значение коэф. силуэта', fontsize=14)
axes[0].set_xlabel('Количество кластеров', fontsize=14)
axes[0].set_xticks(K)
axes[0].grid(True)

optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
axes[0].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, alpha=0.7)
axes[0].text(optimal_k + 0.5, max(silhouette_scores) * 0.95, f'Оптимальное k = {optimal_k}', color='r', fontsize=12);

silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_4)
    score = silhouette_score(X_4, kmeans.labels_)
    silhouette_scores.append(score)

axes[1].plot(K, silhouette_scores, 'bo-', markersize=8, linewidth=2, alpha=0.7)
axes[1].set_ylabel('Среднее значение коэф. силуэта', fontsize=14)
axes[1].set_xlabel('Количество кластеров', fontsize=14)
axes[1].set_xticks(K)
axes[1].grid(True)

optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
axes[1].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, alpha=0.7)
axes[1].text(optimal_k + 0.5, max(silhouette_scores) * 0.95, f'Оптимальное k = {optimal_k}', color='r', fontsize=12);

plt.tight_layout()
plt.savefig('file3.png')
plt.show()