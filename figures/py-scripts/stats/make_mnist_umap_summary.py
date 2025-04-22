import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define the dimensions and scores
dim = ['h=2', 'h=4', 'h=8', 'h=16']

# Adjusted Rand Index (ARI) Scores

km_ari = [0.8154, 0.7971 , 0.7985, 0.8160]
em_ari = [0.8670, 0.8828, 0.8494, 0.8359]
wc_ari = [0.9008, 0.8015, 0.8004, 0.8105]
git_ari = [0.9141, 0.9298, 0.9297, 0.9289]
sb_ari = [0.9304, 0.9318, 0.9315, 0.9332]

# Normalized Mutual Information (NMI) Scores
km_nmi = [0.8664, 0.8586, 0.8574, 0.8641]
em_nmi = [0.8908, 0.9025, 0.8916, 0.8826]
wc_nmi = [0.9047, 0.8733, 0.8721, 0.8785]
git_nmi = [0.9081, 0.9172, 0.9183, 0.9184]
sb_nmi = [0.9182, 0.9198, 0.9125, 0.9214]

# Define colors for each method and their lighter shades
colors = {
    'KM': ('deeppink', 'pink'),
    'EM': ('green', 'lightgreen'),
    'WC': ('red', 'lightcoral'),
    'GIT': ('royalblue', 'lightblue'),
    'SB': ('purple', 'plum')
}

# Helper function to determine the color of each bar
def get_bar_color(score, max_score, method):
    return colors[method][0] if score == max_score else colors[method][1]

# Create the figure with subplots for ARI and NMI
fig = make_subplots(rows=1, cols=2)

# Methods and their corresponding scores
methods = ['KM', 'EM', 'WC', 'GIT', 'SB']
ari_scores = [km_ari, em_ari, wc_ari, git_ari, sb_ari]
nmi_scores = [km_nmi, em_nmi, wc_nmi, git_nmi, sb_nmi]

# Plot bars for each method for ARI scores
for i, (method, score) in enumerate(zip(methods, ari_scores)):
    max_scores = [max(km_ari[i], em_ari[i], wc_ari[i], git_ari[i], sb_ari[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=True  # Only show legend once
    ), row=1, col=1)

# Plot bars for each method for NMI scores
for i, (method, score) in enumerate(zip(methods, nmi_scores)):
    max_scores = [max(km_nmi[i], em_nmi[i], wc_nmi[i], git_nmi[i], sb_nmi[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=False  # Only show legend once
    ), row=1, col=2)

# Update layout with larger and bold axis titles and tick labels
fig.update_layout(
    title={
        'text': "MNIST - UMAP",
        'x': 0.5,  # Center the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 36,  # Make the title font larger
        }
    },
    plot_bgcolor='whitesmoke',
    width=1200,
    height=600,
    xaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    yaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    barmode='group',
    legend=dict(
        font=dict(
            size=20,
            weight='bold'
        ),
        title_font_size=24,
        itemsizing='constant'
    )
)

fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=22))
fig.update_yaxes(title_text='ARI Score', row=1, col=1, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='NMI Score', row=1, col=2, title_font=dict(size=28), tickfont=dict(size=24))

# Save the figure as a pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, "mnist_umap_summary.pdf")
