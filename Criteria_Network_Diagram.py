# ============================================================
# Correlation Network + Hierarchical Clustering Dendrogram
# For an Excel sheet: col1 = Parcel ID, next cols = criteria
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import networkx as nx
import textwrap

def wrap_label(text, width=10):
    return "\n".join(textwrap.wrap(text, width=width))
# -----------------------
# 1) Load Excel
# -----------------------
# Change these:
excel_path = r".xlsx"       
sheet_name = 0                        

df = pd.read_excel(excel_path, sheet_name=sheet_name)
criteria_df = df.iloc[:, 1:].copy()
criteria_df = criteria_df.apply(pd.to_numeric, errors="coerce")
criteria_df = criteria_df.apply(lambda s: s.fillna(s.median()), axis=0)
criterion_type = {
    "Population Density": "benefit",
    "Household Size": "benefit",
    "Sex Ratio": "benefit",
    "Agricultural Labour": "benefit",
    "Flood Extent": "benefit",
    "Settlement Density": "benefit",
    
    "Literacy Rate": "cost",
    "Drinking Water": "cost",
    "Sanitation Facility": "cost",
    "Power Supply": "cost",
    "Communication": "cost",
    "Shelter": "cost",
    "Schools": "cost",
    "Health Facility": "cost",  
    "Employment Rate": "cost",
    "Financial Institution": "cost",
    "Polder": "cost"
}
# -----------------------
# Descriptive statistics: Mean, SD, CV
# -----------------------

# Mean and standard deviation
mean_vals = criteria_df.mean()
std_vals  = criteria_df.std(ddof=0)   # population SD (consistent with normalization)

# Coefficient of Variation (CV)
cv_vals = std_vals / mean_vals

# Combine into a single table
cv_table = pd.DataFrame({
    "Mean": mean_vals,
    "Std_Dev": std_vals,
    "CV": cv_vals
})

print("\nCoefficient of Variation (CV) by criterion:")
print(cv_table)


# -----------------------
# 2) Correlation matrix
# -----------------------
method = "spearman"   # "pearson" or "spearman"
corr = criteria_df.corr(method=method)

# -----------------------
# 3) Correlation Network
# -----------------------

thr = 0.3  

# Build graph
G = nx.Graph()
vars_ = corr.columns.tolist()
G.add_nodes_from(vars_)

# Add edges based on threshold
for i in range(len(vars_)):
    for j in range(i + 1, len(vars_)):
        r = corr.iloc[i, j]
        if np.isfinite(r) and abs(r) >= thr:
            G.add_edge(vars_[i], vars_[j], weight=abs(r), sign=np.sign(r))

# Plot network
plt.figure(figsize=(8, 6))

if G.number_of_edges() == 0:
    print(f"No edges found at threshold |r| >= {thr}. Lower 'thr' and rerun.")
else:
    # Layout
    pos = nx.spring_layout(G, seed=36, k=2.0)

    # Edge widths scale with correlation magnitude
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = 1 + 6 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)

    # Edge colors: positive vs negative correlation
    edge_colors = ["#1f77b4" if G[u][v]["sign"] > 0 else "#d62728" for u, v in G.edges()]

    nx.draw_networkx_nodes(
    G, pos,
    node_size=5000,
    node_color="#9CFF6649",   # light yellow
    edgecolors="black",
    linewidths=1.2
    )
    wrapped_labels = {n: wrap_label(n, width=10) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=wrapped_labels,
        font_size=12,
        font_weight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.7,
        boxstyle="round,pad=0.15")
    )
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8)

    plt.title("(b)",loc="left", fontsize=14, fontweight="bold")         #f"Correlation Network ({method.title()}), threshold |r| ≥ {thr}")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        "correlation_network.jpg",
        dpi=600,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.02)
    plt.show()
    plt.close()
# -----------------------
# 4) Hierarchical clustering dendrogram (variables)
# -----------------------
# Convert correlation to a distance matrix
# Common choice: d = 1 - |r|
dist = 1 - np.abs(corr.values)

# dist must be condensed (upper triangle) for linkage
condensed = squareform(dist, checks=False)

# Linkage method: 'average', 'complete', 'ward' (ward needs Euclidean distances)
Z = linkage(condensed, method="average")

plt.figure(figsize=(6, 5))
dendrogram(
    Z,
    labels=vars_,
    leaf_rotation=90,
    leaf_font_size=13
)
plt.title("(b)", fontsize=14, fontweight="bold")    #f"Hierarchical Clustering of Criteria (distance = 1 - |{method} r|)")
plt.ylabel("Distance",fontsize=14)
plt.tight_layout()

plt.savefig(
    "Hierarchical Clustering.jpg",
    dpi=600,
    format="jpg",
    bbox_inches="tight",
    pad_inches=0.02)
plt.show()

# -----------------------
# 5) Boxplot of criteria (Descriptive visualization)
# -----------------------
out_dir = r""
os.makedirs(out_dir, exist_ok=True)

boxplot_path = os.path.join(out_dir, f"criteria_boxplot_{method}_600dpi.jpg")

# (A) Standardize (z-score) for comparability
crit_plot = (criteria_df - criteria_df.mean()) / criteria_df.std(ddof=0)

# (B) Min-max normalize
eps = 1e-9
for col in crit_plot.columns:
    ctype = criterion_type.get(col)
    if ctype is None:
        raise ValueError(f"Criterion type not defined for: {col}")

    col_min = crit_plot[col].min()
    col_max = crit_plot[col].max()
    denom = (col_max - col_min) + eps

    if ctype.lower() == "benefit":
        crit_plot[col] = (crit_plot[col] - col_min) / denom
    elif ctype.lower() == "cost":
        crit_plot[col] = (col_max - crit_plot[col]) / denom
    else:
        raise ValueError("Criterion type must be 'benefit' or 'cost'")


# -----------------------
# (D)Linear ratio normalization (benefit / cost)
# -----------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    [crit_plot[col].values for col in crit_plot.columns],
    labels=crit_plot.columns,
    patch_artist=True,     
    showfliers=True,      
    widths=0.6
)

for box in bp["boxes"]:
    box.set(facecolor="none",   edgecolor="blue",     linewidth=1.5,linestyle="-")

for median in bp["medians"]:
    median.set(color="black",linewidth=2.0,linestyle="-")

for whisker in bp["whiskers"]:
    whisker.set(color="black",linewidth=1.2,linestyle="--"        )

for cap in bp["caps"]:
    cap.set(color="black",linewidth=1.2,linestyle="-")

for flier in bp["fliers"]:
    flier.set(
        marker="o",            # "o", "s", "^", "D", "+"
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=4,
        alpha=0.6
    )

ax.set_title("Distribution of Vulnerability Criteria", fontsize=14)
ax.set_ylabel("Normalized Value", fontsize=14)
ax.set_xlabel("Criteria", fontsize=10)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)

ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()

# -----------------------
# Save as 600 DPI JPG
# -----------------------
fig.savefig(
    boxplot_path,
    dpi=600,
    format="jpg",
    bbox_inches="tight",
    pad_inches=0.02
)

plt.show()
plt.close(fig)

print("Saved boxplot to:", boxplot_path)
