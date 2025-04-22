import numpy as np
import matplotlib.pyplot as plt
from spectralbridges import SpectralBridges

from sklearn.metrics import adjusted_rand_score

np.random.seed(0)

data = np.genfromtxt('datasets/smile.csv', delimiter=',')

X, y = data[:, :-1], data[:, -1]

k = 4

net = SpectralBridges(n_clusters=k, n_nodes=100)

net.fit(X)

colors = plt.cm.tab10(np.arange(k))

guess = net.predict(X)

for i in range(k):
    plt.scatter(X[(y == i), 0], X[(y == i), 1], color='black', s=.5)
    plt.scatter(net.cluster_centers_[i][:, 0], net.cluster_centers_[i][:, 1], color=colors[i], label=f'Cluster {i}')

plt.show()

ari = adjusted_rand_score(y, guess)

print(f"Adjusted Rand Index: {ari}")
