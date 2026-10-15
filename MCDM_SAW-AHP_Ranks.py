import numpy as np
import pandas as pd
from scipy.stats import rankdata

file_path = r"C:\Users\Swarna\Desktop\PhD\Analysis\Part 1\Python\Sensitivity1y - old (for discover hazard).xlsx"
output_file = r"C:\Users\Swarna\Desktop\PhD\Analysis\Part 1\Python\SAW_Scores_Ranks.xlsx"

df = pd.read_excel(file_path)
unit_col = df.columns[0]

def saw(data, weights, criteria):
    X = data.to_numpy(float)
    N = np.zeros_like(X)

    for j, c in enumerate(criteria):
        mn, mx = np.nanmin(X[:, j]), np.nanmax(X[:, j])
        if np.isclose(mx, mn):
            N[:, j] = 0
        elif c == "benefit":
            N[:, j] = (X[:, j] - mn) / (mx - mn)
        else:
            N[:, j] = (mx - X[:, j]) / (mx - mn)

    scores = N @ np.asarray(weights)
    ranks = rankdata(-scores, method="ordinal").astype(int)
    return scores, ranks

S_cols = ["C1","C2","C3","C4","C5","C6"]
V_cols = ["C7","C8","C9","C10","C11","C12","C13","C14","C15","C16"]
H_cols = ["C17"]

S_types = ["benefit","cost","cost","cost","cost","benefit"]
V_types = ["benefit","benefit","cost","cost","cost","cost","cost","cost","benefit","cost"]
H_types = ["benefit"]

S_w = [0.1169,0.2345,0.1530,0.1698,0.2048,0.1210]
V_w = [0.1204,0.0827,0.0596,0.1605,0.1505,0.0801,0.0767,0.0930,0.0968,0.0797]
H_w = [1.0]
R_w = [0.0292,0.0586,0.0382,0.0425,0.0512,0.0303,0.0301,0.0207,
        0.0149,0.0401,0.0376,0.0200,0.0192,0.0233,0.0242,0.0199,0.5000]

#without CR>0.1
# S_w = [0.0924,0.2000,0.1277,0.1900,0.2717,0.1181]
# V_w = [0.1499,0.0872,0.0669,0.1248,0.1296,0.0867,0.0779,0.0962,0.0919,0.0890]
# H_w = [1.0]
# R_w = [0.0231,0.0500,0.0319,0.0475,0.0679,0.0295,0.0375,0.0218,0.0167,0.0312,
#        0.0324,0.0217,0.0195,0.0241,0.0230,0.0223,0.5000]

S_score, S_rank = saw(df[S_cols], S_w, S_types)
V_score, V_rank = saw(df[V_cols], V_w, V_types)
H_score, H_rank = saw(df[H_cols], [1.0], H_types)
R_score, R_rank = saw(df[S_cols+V_cols+H_cols], R_w, S_types+V_types+H_types)

out = pd.DataFrame({
    "Administrative_Unit": df[unit_col],
    "Susceptibility_Score": S_score, "Susceptibility_Rank": S_rank,
    "Vulnerability_Score": V_score, "Vulnerability_Rank": V_rank,
    "Hazard_Score": H_score, "Hazard_Rank": H_rank,
    "Overall_Risk_Score": R_score, "Overall_Risk_Rank": R_rank
})

out.sort_values("Overall_Risk_Rank").to_excel(output_file, index=False)
print("Saved:", output_file)