import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_score_vs_position(
    df: pd.DataFrame,
    position_col: str,
    score_col: str,
    title: str = "Mean Predicted Score vs Position",
    ylim: tuple = (0, 1),
) -> plt.Figure:
    plot_df = (
        df.groupby(position_col)[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "mean_score", position_col: "position"})
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="position",
        y="mean_score",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Predicted Score")
    ax.set_ylim(ylim)
    sns.despine()
    fig.tight_layout()
    return fig

def plot_ctr_vs_position(
    df: pd.DataFrame,
    position_col: str,
    label_col: str,
    score_col: str,
    title: str = "CTR and Predicted Score vs Position",
    ylim: tuple = (0, 1),
) -> plt.Figure:
    empirical = (
        df.groupby(position_col)[label_col]
        .mean()
        .reset_index()
        .rename(columns={label_col: "mean_score", position_col: "position"})
    )
    empirical["series"] = "empirical_ctr"

    predicted = (
        df.groupby(position_col)[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "mean_score", position_col: "position"})
    )
    predicted["series"] = "predicted_score"

    plot_df = pd.concat([empirical, predicted], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="position",
        y="mean_score",
        hue="series",
        marker="o",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Score / CTR")
    ax.set_ylim(ylim)
    sns.despine()
    fig.tight_layout()
    return fig