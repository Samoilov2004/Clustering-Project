import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_samples = 500
n_features = 2

X_3, y_3 = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, cluster_std=1.5, random_state=42)
X_4, y_4 = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, cluster_std=1.5, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for cluster in range(3):
    axes[0].scatter(X_3[y_3 == cluster, 0], X_3[y_3 == cluster, 1], s=50, c=colors[cluster], label=f'Cluster {cluster + 1}', edgecolors='w', linewidth=0.5)

axes[0].set_xlabel('', fontsize=14)
axes[0].set_ylabel('', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].set_title('График для 3 кластеров', fontsize=16)

for cluster in range(4):
    axes[1].scatter(X_4[y_4 == cluster, 0], X_4[y_4 == cluster, 1], s=50, c=colors[cluster], label=f'Cluster {cluster + 1}', edgecolors='w', linewidth=0.5)

axes[1].set_xlabel('', fontsize=14)
axes[1].set_ylabel('', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].set_title('График для 4 кластеров', fontsize=16)

plt.tight_layout()
plt.savefig('file1.png')
plt.show()