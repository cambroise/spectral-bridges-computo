import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from spectralbridges import SpectralBridges
import git
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Function to load data
def load_data(file_name):
    data = np.genfromtxt(f"datasets/{file_name}", delimiter=",")
    X, y = data[:, :-1], data[:, -1]
    return X, y


# List of files
files = ["impossible.csv", "moons.csv", "circles.csv", "smile.csv"]

# List of parameters

params = {
    "impossible.csv": {
        "DBSCAN": {"eps": 0.5, "min_samples": 8},
        "SpectralBridges": {"n_nodes": 250},
    },
    "moons.csv": {
        "DBSCAN": {"eps": 0.125, "min_samples": 12},
        "SpectralBridges": {"n_nodes": 12},
    },
    "circles.csv": {
        "DBSCAN": {"eps": 0.1, "min_samples": 5},
        "SpectralBridges": {"n_nodes": 25},
    },
    "smile.csv": {
        "DBSCAN": {"eps": 0.125, "min_samples": 12},
        "SpectralBridges": {"n_nodes": 100},
    },
}

# Number of repetitions
N = 200

# Initialize results dictionary
results = {
    "DBSCAN": {},
    "KMeans": {},
    "GaussianMixture": {},
    "AgglomerativeClustering": {},
    "GIT": {},
    "SpectralBridges": {},
}

# Iterate through each file
for file in files:
    # Load data
    X, y = load_data(file)

    # Number of classes
    n_clusters = np.unique(y).shape[0]

    # Initialize arrays for each algorithm and metric
    dbscan_ari = np.zeros(N)
    dbscan_nmi = np.zeros(N)
    kmeans_ari = np.zeros(N)
    kmeans_nmi = np.zeros(N)
    gm_ari = np.zeros(N)
    gm_nmi = np.zeros(N)
    agg_ari = np.zeros(N)
    agg_nmi = np.zeros(N)
    git_ari = np.zeros(N)
    git_nmi = np.zeros(N)
    sb_ari = np.zeros(N)
    sb_nmi = np.zeros(N)

    for i in range(N):
        # DBSCAN
        dbscan = DBSCAN(
            eps=params[file]["DBSCAN"]["eps"],
            min_samples=params[file]["DBSCAN"]["min_samples"],
        )
        y_pred_dbscan = dbscan.fit_predict(X)
        dbscan_ari[i] = adjusted_rand_score(y, y_pred_dbscan)
        dbscan_nmi[i] = normalized_mutual_info_score(y, y_pred_dbscan)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=i)
        y_pred_kmeans = kmeans.fit_predict(X)
        kmeans_ari[i] = adjusted_rand_score(y, y_pred_kmeans)
        kmeans_nmi[i] = normalized_mutual_info_score(y, y_pred_kmeans)

        # Gaussian Mixture
        gm = GaussianMixture(n_components=n_clusters, random_state=i)
        y_pred_gm = gm.fit_predict(X)
        gm_ari[i] = adjusted_rand_score(y, y_pred_gm)
        gm_nmi[i] = normalized_mutual_info_score(y, y_pred_gm)

        # Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred_agg = agg_clustering.fit_predict(X)
        agg_ari[i] = adjusted_rand_score(y, y_pred_agg)
        agg_nmi[i] = normalized_mutual_info_score(y, y_pred_agg)

        # GIT Clustering
        git_clustering = git.GIT(k=40, target_ratio=[1 for _ in range(n_clusters)])
        y_pred_git = git_clustering.fit_predict(X)
        git_ari[i] = adjusted_rand_score(y, y_pred_git)
        git_nmi[i] = normalized_mutual_info_score(y, y_pred_git)

        # Spectral-Bridges
        sb = SpectralBridges(
            n_clusters=n_clusters,
            n_nodes=params[file]["SpectralBridges"]["n_nodes"],
            random_state=i,
        )
        sb.fit_select(X)
        y_pred_sb = sb.predict(X)
        sb_ari[i] = adjusted_rand_score(y, y_pred_sb)
        sb_nmi[i] = normalized_mutual_info_score(y, y_pred_sb)

        print(f"File : {file}, {int((100 * i) / N)}% done")

    # Store results in dictionary
    results["DBSCAN"][file] = {"ARI": dbscan_ari, "NMI": dbscan_nmi}

    results["KMeans"][file] = {"ARI": kmeans_ari, "NMI": kmeans_nmi}

    results["GaussianMixture"][file] = {"ARI": gm_ari, "NMI": gm_nmi}

    results["AgglomerativeClustering"][file] = {"ARI": agg_ari, "NMI": agg_nmi}

    results["GIT"][file] = {"ARI": git_ari, "NMI": git_nmi}

    results["SpectralBridges"][file] = {"ARI": sb_ari, "NMI": sb_nmi}

# Print results, to make a table later on
print()
for method in results:
    for file in files:
        print("ARI", method, file, results[method][file]["ARI"].mean())
        print("NMI", method, file, results[method][file]["NMI"].mean())

# Define files, methods, and metrics
file_labels = ["Impossible", "Moons", "Circles", "Smile"]
methods = [
    "DBSCAN",
    "KMeans",
    "GaussianMixture",
    "AgglomerativeClustering",
    "GIT",
    "SpectralBridges",
]
metrics = ["ARI", "NMI"]

# Define colors for each method
method_colors = {
    "DBSCAN": "orange",
    "KMeans": "deeppink",
    "GaussianMixture": "green",
    "AgglomerativeClustering": "red",
    "GIT": "royalblue",
    "SpectralBridges": "purple",
}

# Define method labels for the legend
method_labels = {
    "DBSCAN": "DB",
    "KMeans": "KM",
    "GaussianMixture": "EM",
    "AgglomerativeClustering": "WC",
    "GIT": "GIT",
    "SpectralBridges": "SB",
}

# Create boxplots for ARI and NMI
fig = make_subplots(
    rows=len(files),  # Number of rows equals the number of files
    cols=2,  # Two columns for ARI and NMI
    shared_xaxes=True,
    subplot_titles=("ARI", "NMI"),
    vertical_spacing=0.05,  # Reduce vertical spacing
    horizontal_spacing=0.075,  # Reduce horizontal spacing
)

# Helper function to add traces
def add_traces(fig, metric, col):
    for file in files:
        for method in methods:
            data = results[method][file][metric]
            if data.size > 0:
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=method_labels[method],  # Use method labels for legend
                        boxmean="sd",
                        marker=dict(
                            color=method_colors[method]
                        ),  # Assign color based on method
                        text=[method_labels[method]]
                        * len(data),  # Text annotation for each box
                        hoverinfo="text",
                    ),
                    row=files.index(file) + 1,
                    col=col,  # Row index starts from 1
                )


# Add traces for ARI and NMI
for col, metric in enumerate(metrics, start=1):
    add_traces(fig, metric, col)

# Update layout
for i, file in enumerate(file_labels, start=1):
    fig.update_yaxes(title_text=file, row=i, col=1)
    fig.update_xaxes(
        title_text="", row=i, col=1
    )  # Remove x-axis label for better space utilization

fig.update_layout(
    plot_bgcolor='whitesmoke',
    height=1000,
    width=1200,
    title_text="",
    showlegend=False,
    legend_title_text="Methods",
    font=dict(size=20),  # Increase font size
)

for i in range(2):
    fig.layout.annotations[i].font.size = 25  # Increase subplot titles font size


# Save the figure as an pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, "synthetic_summary.pdf")
