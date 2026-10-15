import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import spearmanr
from matplotlib.ticker import PercentFormatter

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("C:\\Users\\Swarna\\Desktop\\PhD\\Analysis\\Part 1\\Python\\AHP_uncertainty\\AHP_participant matrices.xlsx").resolve().parent
EXCEL_PATH = BASE_DIR / "AHP_participant matrices.xlsx"
N_EXPERTS, N_BOOTSTRAP, RANDOM_SEED = 17, 10000, 42

ANALYSES = {
    "Vulnerability": {"sheet_name": 0, "n_criteria": 10},
    "Susceptibility": {"sheet_name": 1, "n_criteria": 6}
}

# Experts with individual CR > 0.10
CR_EXCLUDE_EXPERTS = {
    "Vulnerability": [3, 4, 5, 7, 13],
    "Susceptibility": [2, 3, 5, 9, 14, 15, 16, 17]
}

OUTPUT_PATH = BASE_DIR / "AHP_LOEO_Bootstrap_CR_Sensitivity.xlsx"

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Excel file not found:\n{EXCEL_PATH}")


# ============================================================
# 1. READ EXPERT MATRICES
# ============================================================

def read_expert_matrices(excel_path, sheet_name, n_criteria, n_experts=17):
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    criteria = raw.iloc[0, 1:1+n_criteria].astype(str).str.strip().tolist()
    matrices = []

    for expert in range(n_experts):
        start = 1 + expert * (n_criteria + 1)
        matrix = raw.iloc[start:start+n_criteria, 1:1+n_criteria].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        if np.isnan(matrix).any():
            raise ValueError(f"Missing/non-numeric values for Expert {expert+1}, sheet {sheet_name}")
        if np.any(matrix <= 0):
            raise ValueError(f"Zero/negative values for Expert {expert+1}, sheet {sheet_name}")

        matrices.append(matrix)

    return criteria, np.array(matrices)


# ============================================================
# 2. BASIC AHP FUNCTIONS
# ============================================================

def rgmm_weights(matrix):
    row_gm = np.exp(np.mean(np.log(matrix), axis=1))
    return row_gm / row_gm.sum()


def aggregate_expert_matrices(matrices):
    return np.exp(np.mean(np.log(matrices), axis=0))


def evm_weights(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[idx].real
    vector = np.abs(eigenvectors[:, idx].real)
    return vector / vector.sum(), lambda_max


def evm_weight_error(matrix, weights, lambda_max):
    n = len(weights)
    return np.array([
        np.sqrt(np.sum(((n/lambda_max)*matrix[i, :]*weights - weights[i])**2)/(n-1))
        for i in range(n)
    ])


def modified_cr(lambda_max, n):
    denominator = 2.7699*n - 4.3513 - n
    return (lambda_max-n)/denominator if denominator > 0 else np.nan


def rank_weights(weights):
    return pd.Series(weights).rank(
        method="min", ascending=False
    ).astype(int).to_numpy()


# ============================================================
# 3. PARTICIPANT-LEVEL RGMM DISTRIBUTIONS
# ============================================================

def participant_weight_analysis(matrices, criteria):
    individual = np.array([rgmm_weights(m) for m in matrices])
    df = pd.DataFrame(
        individual,
        index=[f"Expert_{i+1:02d}" for i in range(len(matrices))],
        columns=criteria
    )

    distribution = pd.DataFrame({
        "Mean": df.mean(),
        "SD": df.std(ddof=1),
        "Median": df.median(),
        "Q1": df.quantile(0.25),
        "Q3": df.quantile(0.75),
        "Min": df.min(),
        "Max": df.max()
    })

    return df, distribution


# ============================================================
# 4. LEAVE-ONE-EXPERT-OUT ANALYSIS
# ============================================================

def leave_one_expert_out(matrices, criteria):
    k = len(matrices)

    full_matrix = aggregate_expert_matrices(matrices)
    full_weights, full_lambda = evm_weights(full_matrix)
    full_rank = rank_weights(full_weights)
    full_cr = modified_cr(full_lambda, len(criteria))
    full_error = evm_weight_error(full_matrix, full_weights, full_lambda)

    loo_weights, loo_ranks, rhos, loo_cr = [], [], [], []

    for omitted in range(k):
        remaining = np.delete(matrices, omitted, axis=0)
        weights, lam = evm_weights(aggregate_expert_matrices(remaining))
        ranks = rank_weights(weights)

        loo_weights.append(weights)
        loo_ranks.append(ranks)
        rhos.append(spearmanr(full_weights, weights).statistic)
        loo_cr.append(modified_cr(lam, len(criteria)))

    loo_weights, loo_ranks = np.array(loo_weights), np.array(loo_ranks)
    labels = [f"Without_Expert_{i+1:02d}" for i in range(k)]

    loo_weights_df = pd.concat([
        pd.DataFrame([full_weights], index=["All_Experts"], columns=criteria),
        pd.DataFrame(loo_weights, index=labels, columns=criteria)
    ])

    loo_ranks_df = pd.concat([
        pd.DataFrame([full_rank], index=["All_Experts"], columns=criteria),
        pd.DataFrame(loo_ranks, index=labels, columns=criteria)
    ])

    abs_change = np.abs(loo_weights-full_weights)
    pct_change = abs_change/full_weights*100

    summary = pd.DataFrame({
        "Original_Weight": full_weights,
        "Original_Rank": full_rank,
        "LOO_Mean": loo_weights.mean(axis=0),
        "LOO_SD": loo_weights.std(axis=0, ddof=1),
        "LOO_Min": loo_weights.min(axis=0),
        "LOO_Max": loo_weights.max(axis=0),
        "Max_Absolute_Change": abs_change.max(axis=0),
        "Max_Percent_Change": pct_change.max(axis=0),
        "Minimum_Rank": loo_ranks.min(axis=0),
        "Maximum_Rank": loo_ranks.max(axis=0),
        "Weight_Error_Eq9": full_error
    }, index=criteria)

    summary["Rank_Range"] = summary["Maximum_Rank"]-summary["Minimum_Rank"]
    summary["Weight_Error_Percent"] = full_error/full_weights*100

    omission_summary = pd.DataFrame({
        "Omitted_Expert": [f"Expert_{i+1:02d}" for i in range(k)],
        "Spearman_Rho_with_Full_Ranking": rhos,
        "CR_without_Expert": loo_cr,
        "Maximum_Absolute_Weight_Change": abs_change.max(axis=1),
        "Maximum_Percent_Weight_Change": pct_change.max(axis=1)
    })

    overall = {
        "Full_CR": full_cr,
        "Minimum_Spearman_Rho": np.min(rhos),
        "Mean_Spearman_Rho": np.mean(rhos),
        "Maximum_Spearman_Rho": np.max(rhos)
    }

    return (
        full_weights, full_rank, full_cr,
        loo_weights_df, loo_ranks_df,
        summary, omission_summary, overall
    )


# ============================================================
# 5. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_weights(matrices, criteria, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    k, n = len(matrices), len(criteria)

    results = np.zeros((n_bootstrap, n))
    ranks = np.zeros((n_bootstrap, n), dtype=int)

    for b in range(n_bootstrap):
        sample = matrices[rng.choice(k, size=k, replace=True)]
        weights, _ = evm_weights(aggregate_expert_matrices(sample))
        results[b] = weights
        ranks[b] = rank_weights(weights)

    lower = np.percentile(results, 2.5, axis=0)
    upper = np.percentile(results, 97.5, axis=0)

    summary = pd.DataFrame({
        "Bootstrap_Mean": results.mean(axis=0),
        "Bootstrap_Median": np.median(results, axis=0),
        "Bootstrap_SD": results.std(axis=0, ddof=1),
        "CI_2.5%": lower,
        "CI_97.5%": upper,
        "CI_Width": upper-lower,
        "Median_Rank": np.median(ranks, axis=0),
        "Minimum_Rank": ranks.min(axis=0),
        "Maximum_Rank": ranks.max(axis=0)
    }, index=criteria)

    return summary, pd.DataFrame(results, columns=criteria)


# ============================================================
# 6. CR > 0.10 EXCLUSION SENSITIVITY ANALYSIS
# ============================================================

def cr_exclusion_analysis(
    matrices, criteria, analysis_name,
    full_weights, full_rank, full_cr
):
    excluded = CR_EXCLUDE_EXPERTS[analysis_name]
    keep = [i for i in range(len(matrices)) if i+1 not in excluded]
    filtered_matrices = matrices[keep]

    filtered_matrix = aggregate_expert_matrices(filtered_matrices)
    filtered_weights, filtered_lambda = evm_weights(filtered_matrix)
    filtered_rank = rank_weights(filtered_weights)
    filtered_cr = modified_cr(filtered_lambda, len(criteria))

    rho, rho_p = spearmanr(full_weights, filtered_weights)
    weight_change = filtered_weights-full_weights
    abs_weight_change = np.abs(weight_change)
    pct_weight_change = abs_weight_change/full_weights*100
    rank_change = filtered_rank-full_rank

    comparison = pd.DataFrame({
        "Criterion": criteria,
        "All_Experts_Weight": full_weights,
        "All_Experts_Rank": full_rank,
        "CR_Filtered_Weight": filtered_weights,
        "CR_Filtered_Rank": filtered_rank,
        "Weight_Change": weight_change,
        "Absolute_Weight_Change": abs_weight_change,
        "Percent_Weight_Change": pct_weight_change,
        "Rank_Change": rank_change,
        "Absolute_Rank_Change": np.abs(rank_change)
    })

    summary = pd.DataFrame({
        "Measure": [
            "Analysis",
            "Total experts",
            "Excluded experts",
            "Number excluded",
            "Retained experts",
            "Number retained",
            "Original group CR",
            "CR-filtered group CR",
            "Spearman rho",
            "Spearman p-value",
            "Maximum absolute weight change",
            "Maximum percent weight change",
            "Maximum absolute rank change"
        ],
        "Value": [
            analysis_name,
            len(matrices),
            ", ".join(map(str, excluded)),
            len(excluded),
            ", ".join(str(i+1) for i in keep),
            len(keep),
            full_cr,
            filtered_cr,
            rho,
            rho_p,
            abs_weight_change.max(),
            pct_weight_change.max(),
            np.abs(rank_change).max()
        ]
    })

    return comparison, summary


# ============================================================
# 7. RUN BOTH VULNERABILITY AND SUSCEPTIBILITY
# ============================================================

all_results = {}

for analysis_name, settings in ANALYSES.items():

    print("\n"+"="*70)
    print("Running:", analysis_name)
    print("="*70)

    criteria, matrices = read_expert_matrices(
        EXCEL_PATH,
        settings["sheet_name"],
        settings["n_criteria"],
        N_EXPERTS
    )

    print("Criteria:", criteria)
    print("Matrices loaded:", matrices.shape)

    individual_weights, participant_distribution = participant_weight_analysis(
        matrices, criteria
    )

    (
        full_weights, full_rank, full_cr,
        loo_weights, loo_ranks, loo_summary,
        omission_summary, overall
    ) = leave_one_expert_out(matrices, criteria)

    bootstrap_summary, bootstrap_raw = bootstrap_weights(
        matrices, criteria,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED
    )

    bootstrap_summary.insert(0, "Original_Weight", full_weights)
    bootstrap_summary.insert(1, "Original_Rank", full_rank)

    cr_compare, cr_summary = cr_exclusion_analysis(
        matrices, criteria, analysis_name,
        full_weights, full_rank, full_cr
    )

    all_results[analysis_name] = {
        "individual_weights": individual_weights,
        "participant_distribution": participant_distribution,
        "loo_weights": loo_weights,
        "loo_ranks": loo_ranks,
        "loo_summary": loo_summary,
        "omission_summary": omission_summary,
        "bootstrap_summary": bootstrap_summary,
        "bootstrap_raw": bootstrap_raw,
        "full_cr": full_cr,
        "overall": overall,
        "cr_compare": cr_compare,
        "cr_summary": cr_summary
    }

    print("\nOriginal group CR:", round(full_cr, 4))

    print("\nOriginal weights:")
    for criterion, weight, rank in zip(criteria, full_weights, full_rank):
        print(f"{criterion:30s} {weight:.5f}  Rank={rank}")

    print("\nLeave-one-out rank stability:")
    print("Minimum Spearman rho:", round(overall["Minimum_Spearman_Rho"], 4))
    print("Mean Spearman rho:", round(overall["Mean_Spearman_Rho"], 4))

    print("\nCR > 0.10 exclusion sensitivity:")
    print(cr_summary.to_string(index=False))


# ============================================================
# 8. PLOT: COMBINED VULNERABILITY + SUSCEPTIBILITY
# EXACT ORIGINAL PLOT STYLE
# ============================================================

def draw_weight_panel(
    ax,
    individual_weights,
    loo_summary,
    bootstrap_summary,
    analysis_name
):
    criteria = individual_weights.columns.tolist()

    final_weights = loo_summary.loc[
        criteria,
        "Original_Weight"
    ]

    order = final_weights.sort_values(
        ascending=True
    ).index.tolist()

    individual_weights = individual_weights[order]
    loo_summary = loo_summary.loc[order]
    bootstrap_summary = bootstrap_summary.loc[order]

    y = np.arange(len(order))

    # =========================================================
    # 1. INDIVIDUAL EXPERT WEIGHTS
    # =========================================================

    for idx, criterion in enumerate(order):

        expert_values = individual_weights[
            criterion
        ].values

        rng = np.random.default_rng(100 + idx)

        jitter = rng.uniform(
            -0.10,
            0.10,
            size=len(expert_values)
        )

        ax.scatter(
            expert_values,
            np.full(len(expert_values), idx) + jitter,
            s=22,
            alpha=0.35,
            marker="o",
            label=(
                "Individual expert weights"
                if idx == 0
                else None
            ),
            zorder=2
        )

    # =========================================================
    # 2. LEAVE-ONE-EXPERT-OUT RANGE
    # =========================================================

    loo_min = loo_summary[
        "LOO_Min"
    ].values

    loo_max = loo_summary[
        "LOO_Max"
    ].values

    for idx in range(len(order)):

        ax.hlines(
            y=idx,
            xmin=loo_min[idx],
            xmax=loo_max[idx],
            linewidth=12,
            alpha=0.45,
            label=(
                "Leave-one-expert-out range"
                if idx == 0
                else None
            ),
            zorder=3
        )

        ax.vlines(
            loo_min[idx],
            idx - 0.08,
            idx + 0.08,
            linewidth=1
        )

        ax.vlines(
            loo_max[idx],
            idx - 0.08,
            idx + 0.08,
            linewidth=1
        )

    # =========================================================
    # 3. 95% BOOTSTRAP INTERVAL
    # =========================================================

    ci_low = bootstrap_summary[
        "CI_2.5%"
    ].values

    ci_high = bootstrap_summary[
        "CI_97.5%"
    ].values

    for idx in range(len(order)):

        ax.hlines(
            y=idx,
            xmin=ci_low[idx],
            xmax=ci_high[idx],
            linewidth=2,
            label=(
                "95% bootstrap interval"
                if idx == 0
                else None
            ),
            zorder=5
        )

        ax.vlines(
            ci_low[idx],
            idx - 0.12,
            idx + 0.12,
            linewidth=1.2
        )

        ax.vlines(
            ci_high[idx],
            idx - 0.12,
            idx + 0.12,
            linewidth=1.2
        )

    # =========================================================
    # 4. FINAL GROUP WEIGHTS
    # =========================================================

    group_weights = loo_summary[
        "Original_Weight"
    ].values

    # =========================================================
    # 5. EQ. 9 WEIGHT / CONSISTENCY ERROR
    # =========================================================

    weight_error = loo_summary[
        "Weight_Error_Eq9"
    ].values

    err = ax.errorbar(
        group_weights,
        y,
        xerr=weight_error,
        fmt="none",
        ecolor="black",
        elinewidth=2.8,
        capsize=4,
        capthick=2.0,
        zorder=8,
        label="Consistency error"
    )

    for barline in err.lines[2]:
        barline.set_linestyle("dotted")

    # =========================================================
    # 6. FINAL GROUP WEIGHT — BLACK DIAMOND
    # =========================================================

    ax.scatter(
        group_weights,
        y,
        marker="D",
        color="black",
        edgecolor="none",
        linewidth=0.5,
        s=40,
        zorder=10,
        label="Final group weight"
    )

    # =========================================================
    # 7. FINAL WEIGHT LABELS ABOVE DIAMONDS
    # =========================================================

    y_offset = 0.20

    for idx, weight in enumerate(group_weights):

        ax.text(
            weight,
            idx + y_offset,
            f"{weight * 100:.2f}%",
            fontsize=14,
            fontweight="regular",
            ha="center",
            va="bottom",
            zorder=12
        )

    # =========================================================
    # 8. PANEL FORMATTING
    # =========================================================

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=14)
    ax.tick_params(axis="x", labelsize=16)
    ax.xaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Criterion Weight", fontsize=16)
    ax.set_ylabel("")
    ax.set_title(
        analysis_name,
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.35
    )

    ax.set_ylim(
        -0.50,
        len(order) - 0.50 + 0.35
    )


# ============================================================
# 9. CREATE COMBINED TWO-PANEL FIGURE
# EXACT ORIGINAL FORMAT
# ============================================================

def plot_combined_weight_stability(
    all_results,
    output_folder
):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 10.5),
        sharex=False,
        gridspec_kw={
            "height_ratios": [10, 6],
            "hspace": 0.32
        }
    )

    # =========================================================
    # (a) VULNERABILITY
    # =========================================================

    vul = all_results[
        "Vulnerability"
    ]

    draw_weight_panel(
        ax=axes[0],
        individual_weights=vul[
            "individual_weights"
        ],
        loo_summary=vul[
            "loo_summary"
        ],
        bootstrap_summary=vul[
            "bootstrap_summary"
        ],
        analysis_name="(a) Vulnerability"
    )

    # =========================================================
    # (b) SUSCEPTIBILITY
    # =========================================================

    sus = all_results[
        "Susceptibility"
    ]

    draw_weight_panel(
        ax=axes[1],
        individual_weights=sus[
            "individual_weights"
        ],
        loo_summary=sus[
            "loo_summary"
        ],
        bootstrap_summary=sus[
            "bootstrap_summary"
        ],
        analysis_name="(b) Susceptibility"
    )

    # =========================================================
    # COMMON LEGEND
    # =========================================================

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    desired_order = [
        "Individual expert weights",
        "Leave-one-expert-out range",
        "95% bootstrap interval",
        "Final group weight",
        "Consistency error"
    ]

    handle_dict = dict(
        zip(labels, handles)
    )

    legend_handles = [
        handle_dict[label]
        for label in desired_order
        if label in handle_dict
    ]

    legend_labels = [
        label
        for label in desired_order
        if label in handle_dict
    ]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=14
    )

    # =========================================================
    # COMMON FIGURE MARGINS
    # =========================================================

    fig.subplots_adjust(
        left=0.36,
        right=0.97,
        top=0.96,
        bottom=0.17,
        hspace=0.32
    )

    # =========================================================
    # SAVE COMBINED FIGURE
    # =========================================================

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        output_folder,
        "AHP_Expert_Weight_Stability_Combined.png"
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(
        "Saved combined figure:",
        output_path
    )


# ============================================================
# 10. EXPORT EVERYTHING TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl"
) as writer:

    for analysis_name, results in all_results.items():

        prefix = (
            "VUL"
            if analysis_name == "Vulnerability"
            else "SUS"
        )

        # ORIGINAL OUTPUTS — UNCHANGED
        results["individual_weights"].to_excel(
            writer, sheet_name=f"{prefix}_Individual"
        )

        results["participant_distribution"].to_excel(
            writer, sheet_name=f"{prefix}_ParticipantDist"
        )

        results["loo_weights"].to_excel(
            writer, sheet_name=f"{prefix}_LOO_Weights"
        )

        results["loo_ranks"].to_excel(
            writer, sheet_name=f"{prefix}_LOO_Ranks"
        )

        results["loo_summary"].to_excel(
            writer, sheet_name=f"{prefix}_LOO_Summary"
        )

        results["omission_summary"].to_excel(
            writer,
            sheet_name=f"{prefix}_LOO_Experts",
            index=False
        )

        results["bootstrap_summary"].to_excel(
            writer, sheet_name=f"{prefix}_Bootstrap_CI"
        )

        results["bootstrap_raw"].to_excel(
            writer,
            sheet_name=f"{prefix}_BootstrapRaw",
            index=False
        )

        # NEW CR-SENSITIVITY OUTPUTS
        results["cr_compare"].to_excel(
            writer,
            sheet_name=f"{prefix}_CR_Compare",
            index=False
        )

        results["cr_summary"].to_excel(
            writer,
            sheet_name=f"{prefix}_CR_Summary",
            index=False
        )


# ============================================================
# 11. ORIGINAL FIGURE
# ============================================================

FIGURE_FOLDER = str(BASE_DIR)

plot_combined_weight_stability(
    all_results,
    FIGURE_FOLDER
)

print("\n"+"="*70)
print("Analysis complete.")
print("Results saved to:")
print(OUTPUT_PATH)
print("="*70)