import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from spectralbridges import SpectralBridges

# Set random seed for reproducibility
np.random.seed(0)

# Load the dataset
data = np.genfromtxt('datasets/impossible.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Parameters for Gaussian noise
mean = 0
std = 0.1 # Standard deviation for the Gaussian noise

# Generate Gaussian noise
gaussian_noise = np.random.normal(mean, std, X.shape)

# Add Gaussian noise to the dataset
X = X + gaussian_noise

# Define the number of clusters
k = 7

# Initialize and fit the SpectralBridges model
net = SpectralBridges(n_clusters=k, n_nodes=250, random_state=42)
net.fit_select(X, n_redo=20)

# Predict cluster labels
guess = net.predict(X)

# Create a plotly figure
fig = go.Figure()

# Add scatter plots for each cluster
for i in range(k):
    fig.add_trace(go.Scatter(
        x=X[(y == i), 0], 
        y=X[(y == i), 1], 
        mode='markers', 
        marker=dict(color='black', size=5), 
    ))
    
for i in range(k):
    fig.add_trace(go.Scatter(
        x=net.cluster_centers_[i][:, 0], 
        y=net.cluster_centers_[i][:, 1], 
        mode='markers', 
        marker=dict(size=15), 
    ))

# Update layout
fig.update_layout(
    plot_bgcolor='whitesmoke',
    xaxis_title="X",
    yaxis_title="Y",
    width=800,
    height=600,
    showlegend=False
)

# Save the figure as an pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, 'gaussian_noise_impossible.pdf')
