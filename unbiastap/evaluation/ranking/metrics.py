import numpy as np
import pandas as pd

def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def _ndcg_for_query(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    ranked_idx = np.argsort(scores)[::-1]
    ranked_labels = labels[ranked_idx].astype(float)

    ideal_idx = np.argsort(labels)[::-1]
    ideal_labels = labels[ideal_idx].astype(float)

    dcg = _dcg_at_k(ranked_labels, k)
    idcg = _dcg_at_k(ideal_labels, k)

    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k(
    df: pd.DataFrame,
    query_col: str,
    label_col: str,
    score_col: str,
    k: int,
) -> float:
    scores_per_query = []

    for _, group in df.groupby(query_col):
        labels = group[label_col].to_numpy()
        scores = group[score_col].to_numpy()
        scores_per_query.append(_ndcg_for_query(labels, scores, k))

    return float(np.mean(scores_per_query))


def _debiased_ndcg_for_query(
    labels: np.ndarray,
    scores: np.ndarray,
    propensities: np.ndarray,
    k: int,
) -> float:
    ips_relevances = labels.astype(float) / np.clip(propensities, 1e-6, None)

    ranked_idx = np.argsort(scores)[::-1]
    ranked_ips = ips_relevances[ranked_idx]

    ideal_idx = np.argsort(ips_relevances)[::-1]
    ideal_ips = ips_relevances[ideal_idx]

    dcg = _dcg_at_k(ranked_ips, k)
    idcg = _dcg_at_k(ideal_ips, k)

    return dcg / idcg if idcg > 0 else 0.0


def debiased_ndcg_at_k(
    df: pd.DataFrame,
    query_col: str,
    label_col: str,
    score_col: str,
    propensity_col: str,
    k: int,
) -> float:
    scores_per_query = []

    for _, group in df.groupby(query_col):
        labels = group[label_col].to_numpy()
        scores = group[score_col].to_numpy()
        propensities = group[propensity_col].to_numpy()
        scores_per_query.append(
            _debiased_ndcg_for_query(labels, scores, propensities, k)
        )

    return float(np.mean(scores_per_query))
