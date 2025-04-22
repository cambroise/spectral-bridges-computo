import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from spectralbridges import SpectralBridges

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data)
y = mnist.target

# Apply PCA to reduce dimensionality
X_reduced = PCA(n_components=32, random_state=42).fit_transform(X)

# Parameters
N = 20  # Number of iterations for averaging ARI and NMI
k = 10  # Number of clusters
n_nodes_range = np.linspace(10, 1000, 10, dtype=int)  # Range of number of nodes
ari_scores = []  # List to store ARI scores
nmi_scores = []  # List to store NMI scores
ari_errors = []  # List to store ARI errors
nmi_errors = []  # List to store NMI errors

# Loop over the range of number of nodes
for n_nodes in n_nodes_range:
    print(f'Number of nodes: {n_nodes}')

    # Compute the average ARI and NMI over N iterations
    ari = np.zeros(N)
    nmi = np.zeros(N)
    for i in range(N):
        if n_nodes > 10:
            net = SpectralBridges(n_clusters=k, n_nodes=int(n_nodes), random_state=i)
            net.fit(X_reduced)
            predictions = net.predict(X_reduced)
        else:
            net = KMeans(n_clusters=k, random_state=i)
            predictions = net.fit_predict(X_reduced)
            
        ari[i] = adjusted_rand_score(y, predictions)
        nmi[i] = normalized_mutual_info_score(y, predictions)

    # Append the average ARI and NMI to the respective lists
    ari_scores.append(ari.mean())
    nmi_scores.append(nmi.mean())
    # Append the standard deviation as error bars
    ari_errors.append(ari.std())
    nmi_errors.append(nmi.std())

# Create subplots
fig = make_subplots(rows=1, cols=2)

# Add ARI plot with error bars
fig.add_trace(go.Scatter(
    x=n_nodes_range, 
    y=ari_scores, 
    mode='lines+markers', 
    name='ARI',
    error_y=dict(type='data', array=ari_errors, visible=True)
), row=1, col=1)

# Add NMI plot with error bars
fig.add_trace(go.Scatter(
    x=n_nodes_range, 
    y=nmi_scores, 
    mode='lines+markers', 
    name='NMI',
    error_y=dict(type='data', array=nmi_errors, visible=True)
), row=1, col=2)

# Update layout
fig.update_layout(
    plot_bgcolor='whitesmoke',
    height=600,
    width=1400,
    showlegend=False,
    xaxis=dict(title='m', titlefont=dict(size=28), tickfont=dict(size=24)),
    yaxis=dict(title='ARI Score', titlefont=dict(size=28), tickfont=dict(size=24)),
    xaxis2=dict(title='m', titlefont=dict(size=28), tickfont=dict(size=24)),
    yaxis2=dict(title='NMI Score', titlefont=dict(size=28), tickfont=dict(size=24)),
)

# Save the figure as an pdf file
pio.kaleido.scope.mathjax = None
fig.write_image("nodes_vs_score.pdf", format='pdf')
