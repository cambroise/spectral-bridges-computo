from spectralbridges import SpectralBridges
import numpy as np
import timeit
import plotly.graph_objs as go
import plotly.io as pio

# Parameters
min_n_nodes = 10
max_n_nodes = 1000
num_subdivisions = 20
N = 50  # Number of runs for each dataset size

# Generate dataset sizes
n_nodes_range = np.linspace(min_n_nodes, max_n_nodes, num_subdivisions, dtype=int)

# Measure and store average time per run for each dataset size
results = []

# Generate sample data
X = np.random.rand(5000, 10)

for n_nodes in n_nodes_range:
    # Initialize Spectral Bridges
    model = SpectralBridges(n_clusters=5, n_nodes=int(n_nodes), random_state=42)

    # Time the execution
    elapsed_time = timeit.timeit(lambda: model.fit(X), number=N)
    avg_time_per_run = elapsed_time / N

    results.append((n_nodes, avg_time_per_run))
    print(f"m: {n_nodes}, Average time per run: {avg_time_per_run:.4f} seconds")

# Extract sizes and average times for plotting
sizes, avg_times = zip(*results)

# Create the plot with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sizes, y=avg_times,
    mode='lines+markers',
    marker=dict(color='blue'),
    line=dict(color='blue'),
    name='Average Time per Run'
))

# Update layout
fig.update_layout(
    plot_bgcolor='whitesmoke',
    xaxis_title='m',
    yaxis_title='Average Time per Run (seconds)',
    xaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    yaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    )
)

# Save the plot as a pdf
pio.kaleido.scope.mathjax = None
pio.write_image(fig, 'nodes_vs_time.pdf')
