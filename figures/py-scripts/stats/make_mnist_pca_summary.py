import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define the dimensions and scores
dim = ['h=8', 'h=16', 'h=32', 'h=64', 'h=784 (full)']

# Adjusted Rand Index (ARI) Scores
km_ari = [0.3661, 0.3701, 0.3794, 0.3740, 0.3665]
em_ari = [0.4672, 0.5081, 0.4826, 0.4422, 0.1850]
wc_ari = [0.4123, 0.4729, 0.4771, 0.5043, 0.5164]
git_ari = [0.4162, 0.4792, 0.3166, 0.2347, 0.3094]
sb_ari = [0.5789, 0.6875, 0.7110, 0.6983, 0.6619]

# Normalized Mutual Information (NMI) Scores
km_nmi = [0.4778, 0.4899, 0.5037, 0.4969, 0.4915]
em_nmi = [0.6007, 0.6462, 0.6355, 0.5952, 0.3252]
wc_nmi = [0.5471, 0.6340, 0.6532, 0.6634, 0.6648]
git_nmi = [0.5110, 0.5720, 0.4227, 0.3314, 0.4297]
sb_nmi = [0.6592, 0.7616, 0.7895, 0.7846, 0.7628]

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
        'text': "MNIST - PCA",
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
pio.write_image(fig, "mnist_pca_summary.pdf")
