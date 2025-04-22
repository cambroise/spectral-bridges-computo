import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from spectralbridges import SpectralBridges

# Set random seed for reproducibility
np.random.seed(0)

# Load the dataset
data = np.genfromtxt('datasets/impossible.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Function to generate random points within a rectangle
def rect_uniform(x_min, x_max, y_min, y_max, n_samples):
    x_random = np.random.uniform(x_min, x_max, n_samples)
    y_random = np.random.uniform(y_min, y_max, n_samples)
    random_points = np.vstack((x_random, y_random)).T
    return random_points

# Generate uniform noise
n = 250
uniform_noise = rect_uniform(X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max(), n)

# Combine original data with uniform noise
X = np.vstack((X, uniform_noise))
y = np.concatenate((y, np.full(n, -1)))

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
pio.write_image(fig, 'uniform_noise_impossible.pdf')
