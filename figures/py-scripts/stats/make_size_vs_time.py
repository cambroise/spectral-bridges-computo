from spectralbridges import SpectralBridges
import numpy as np
import timeit
import plotly.graph_objs as go
import plotly.io as pio

# Parameters
min_size = 100
max_size = 100000
num_subdivisions = 20
N = 50  # Number of runs for each dataset size

# Generate dataset sizes
sizes = np.linspace(min_size, max_size, num_subdivisions, dtype=int)

# Initialize Spectral Bridges
model = SpectralBridges(n_clusters=5, n_nodes=10, random_state=42)

# Measure and store average time per run for each dataset size
results = []

for size in sizes:
    # Generate sample data
    X = np.random.rand(size, 10)

    # Time the execution
    elapsed_time = timeit.timeit(lambda: model.fit(X), number=N)
    avg_time_per_run = elapsed_time / N

    results.append((size, avg_time_per_run))
    print(f"Dataset size: {size}, Average time per run: {avg_time_per_run:.4f} seconds")

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
    xaxis_title='Dataset Size',
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
pio.write_image(fig, 'size_vs_time.pdf')
