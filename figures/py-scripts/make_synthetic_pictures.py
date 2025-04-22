import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Set random state for reproducibility
random_state = 42

# Load data function
def load_data(file_name):
    data = np.genfromtxt(f"datasets/{file_name}", delimiter=",")[1:]
    X, y = data[:, :-1], data[:, -1]
    return X, y

# Function to create scatter plots
def create_scatter_plot(df, show_legend):
    fig = go.Figure()
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=df_subset['x'],
            y=df_subset['y'],
            mode='markers',
            marker=dict(size=5),
            name=label
        ))
    fig.update_layout(
        plot_bgcolor='whitesmoke',
        showlegend=show_legend,
        legend_title_text='Label',
        legend_traceorder='normal',
        legend=dict(
            traceorder='normal',
            title_font=dict(size=18),
            itemsizing='constant',
            itemwidth=30,
        )
    )
    return fig

# List of datasets and their corresponding filenames
datasets = [
    ('impossible', 'impossible.csv'),
    ('moons', 'moons.csv'),
    ('circles', 'circles.csv'),
    ('smile', 'smile.csv')
]

# Loop through each dataset
for dataset_name, file_name in datasets:
    X, y = load_data(file_name)
    
    # Create dataframes for each plot
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['label'] = y.astype(int).astype(str)

    # Create scatter plots
    fig = create_scatter_plot(df, show_legend=False)
    
    # Save the plots to pdf files
    pio.kaleido.scope.mathjax = None
    pio.write_image(fig, f'{dataset_name}.pdf', format='pdf')

