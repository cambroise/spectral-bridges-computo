import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from spectralbridges import SpectralBridges
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import fetch_openml
import umap
import plotly.graph_objects as go
import plotly.io as pio

# Set random state for reproducibility
random_state = 42

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data)
y = mnist.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=32, random_state=42)
X_pca = pca.fit_transform(X)

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=10, random_state=random_state)
y_pred_kmeans = kmeans.fit_predict(X_pca)

# Perform clustering with SpectralBridges
sb = SpectralBridges(n_clusters=10, n_nodes=500, random_state=random_state)
sb.fit_select(X_pca)
y_pred_sb = sb.predict(X_pca)

# Evaluate clustering performance
kmeans_ari = adjusted_rand_score(y, y_pred_kmeans)
kmeans_nmi = normalized_mutual_info_score(y, y_pred_kmeans)
sb_ari = adjusted_rand_score(y, y_pred_sb)
sb_nmi = normalized_mutual_info_score(y, y_pred_sb)

print(f"KMeans ARI: {kmeans_ari}, NMI: {kmeans_nmi}")
print(f"SpectralBridges ARI: {sb_ari}, NMI: {sb_nmi}")

# Transform data with UMAP for visualization
umap_model = umap.UMAP(n_components=2, random_state=random_state)
X_umap = umap_model.fit_transform(X_pca)

# Create dataframes for each plot
df_GTumap = pd.DataFrame(X_umap, columns=['x', 'y'])
df_GTumap['label'] = y

df_SBumap = pd.DataFrame(X_umap, columns=['x', 'y'])
df_SBumap['label'] = y_pred_sb

df_KMumap = pd.DataFrame(X_umap, columns=['x', 'y'])
df_KMumap['label'] = y_pred_kmeans

# Ensure label column is of type string for proper color mapping in plotly
df_GTumap['label'] = df_GTumap['label'].astype(int).astype(str)
df_SBumap['label'] = df_SBumap['label'].astype(int).astype(str)
df_KMumap['label'] = df_KMumap['label'].astype(int).astype(str)

# Sort Ground Truth data by label to ensure legend is ordered
df_GTumap = df_GTumap.sort_values(by='label')

# Function to create scatter plots
def create_scatter_plot(df, show_legend):
    fig = go.Figure()
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=df_subset['x'],
            y=df_subset['y'],
            mode='markers',
            marker=dict(size=2),
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

# Create scatter plots
fig_GTumap = create_scatter_plot(df_GTumap, show_legend=True)
fig_SBumap = create_scatter_plot(df_SBumap, show_legend=False)
fig_KMumap = create_scatter_plot(df_KMumap, show_legend=False)

# Save the plots to pdf files
pio.kaleido.scope.mathjax = None
pio.write_image(fig_GTumap, 'GTumap.pdf', format='pdf')
pio.write_image(fig_SBumap, 'SBumap.pdf', format='pdf')
pio.write_image(fig_KMumap, 'KMumap.pdf', format='pdf')
