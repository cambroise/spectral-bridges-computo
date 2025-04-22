import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from spectralbridges import SpectralBridges
from sklearn.datasets import fetch_openml
import git 
import umap

def evaluate_clustering(X, y, algorithm, n_clusters, n_iterations):
    ari_scores = []
    nmi_scores = []

    for i in range(n_iterations):
        indices = np.random.choice(X.shape[0], 20000, replace=False)
        X_, y_ = X[indices], y[indices]

        match algorithm:
            case "GIT":
                model = git.GIT(k=40, target_ratio=[1 for _ in range(n_clusters)])
                labels = model.fit_predict(X_)
            case "KM":
                model = KMeans(n_clusters=n_clusters, random_state=i)
                labels = model.fit_predict(X_)
            case "EM":
                model = GaussianMixture(n_components=n_clusters, random_state=i)
                labels = model.fit_predict(X_)
            case "WC":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = model.fit_predict(X_)
            case "SB":
                model = SpectralBridges(n_clusters=n_clusters, n_nodes=500, random_state=i)
                model.fit(X_)
                labels = model.predict(X_)
            case _:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        ari_scores.append(adjusted_rand_score(y_, labels))
        nmi_scores.append(normalized_mutual_info_score(y_, labels))

        print(f"{algorithm} iteration : {i}")

    avg_ari = np.mean(ari_scores)
    avg_nmi = np.mean(nmi_scores)
    
    return avg_ari, avg_nmi

np.random.seed(0)

# Step 2: Load MNIST dataset and apply PCA/UMAP
mnist = fetch_openml('mnist_784', version=1)
X0 = np.array(mnist.data)
y = mnist.target

for n in [16, 32, 64]:
    X = PCA(n_components=n, random_state=42).fit_transform(X0)
    #X = umap.UMAP(n_components=n, random_state=42).fit_transform(X0)

    # Step 3: Set clustering parameters
    n_clusters = 10
    n_iterations = 10
    algorithms = ["KM, EM, WC, SB"]

    # Step 4: Evaluate each clustering algorithm and print results
    for algorithm in algorithms:
        avg_ari, avg_nmi = evaluate_clustering(X0, y, algorithm, n_clusters, n_iterations)
        print(f"{algorithm}, {n} - Adjusted Rand Index: {avg_ari:.4f}, Normalized Mutual Information: {avg_nmi:.4f}")
