import math

from sklearn.metrics import roc_auc_score
from custom_types import ItemID, RankingDF, validate_ranking_df


def minrank(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary minrank@k: relevance=1 if item is targeted.
    Lower is better. This metric is 1-indexed.
    If there is no targeted item before position k, then it is treated as nothing present.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: 1-indexed minimum ranking position.
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # identify first relevant item
    relevance = [1 if r else 0 for r in ranking_topk["relevance"]]
    # extract min relevant rank
    minrank = 0
    try:
        minrank = relevance.index(1) + 1
    except ValueError:
        # no relevant items; fallback to NaN
        minrank = float("nan")
    return minrank


def precision(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary precision@k: relevance=1 if item is targeted.
    Higher is better.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: precision@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # return precision
    return ranking_topk["relevance"].sum() / len(ranking_topk)


def recall(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary recall@k: relevance=1 if item is targeted.
    Higher is better.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: recall@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # return recall
    return ranking_topk["relevance"].sum() / ranking["relevance"].sum()


def fscore(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
    beta: float = 1,
) -> float:
    """
    Compute binary fscore@k: relevance=1 if item is targeted.
    Higher is better.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :param beta: beta
    :type beta: float
    :return: fscore@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # retrieve precision and recall
    precision_k = precision(
        ranking=ranking,
        targeted_ids=targeted_ids,
        k=k,
    )
    recall_k = recall(
        ranking=ranking,
        targeted_ids=targeted_ids,
        k=k,
    )
    # compute f-score
    return ((1 + beta * beta) * precision_k * recall_k) / (
        (beta * beta * precision_k) + recall_k
    )


def mrr(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary mrr@k: relevance=1 if item is targeted.
    Higher is better.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: mrr@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # identify first relevant item
    relevance = [1 if r else 0 for r in ranking_topk["relevance"]]
    # mrr is the inverse of that item's position
    mrr = 0
    try:
        mrr = 1 / (relevance.index(1) + 1)
    except ValueError:
        # no relevant items; fallback to 1/infinity ~ 0
        mrr = 0
    return mrr


def ndcg(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary ndcg@k: relevance=1 if item is targeted.
    Higher is better.
    Uses "stricter" log2 NDCG formula, which is fine for binary relevance.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: ndcg@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # compute gains: binary 0/1
    gains = [1 if r else 0 for r in ranking_topk["relevance"]]

    # DCG@k
    dcg = sum((2**g - 1) / math.log2(idx + 2) for idx, g in enumerate(gains))

    # ideal DCG: all relevant items (up to k) at the top
    # ASSUME FULL RE-SORT BEFORE TRUNCATION
    # https://stats.stackexchange.com/questions/341611/proper-way-to-use-ndcgk-score-for-recommendations
    ideal_rel_count = min(sum(ranking["relevance"]), k)
    ideal_gains = [1] * ideal_rel_count
    idcg = sum((2**g - 1) / math.log2(idx + 2) for idx, g in enumerate(ideal_gains))

    return (dcg / idcg) if idcg > 0 else 0.0


def auc(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    k: int,
) -> float:
    """
    Compute binary AUC-ROC@k: relevance=1 if item is targeted.
    Higher is better.

    :param ranking: Ranking
    :type ranking: RankingDF
    :param targeted_ids: Item IDs that are considered targeted
    :type targeted_ids: list[ItemID]
    :param k: k cutoff
    :type k: int
    :return: auc@k
    :rtype: float
    """
    validate_ranking_df(ranking)
    ranking = ranking.copy()

    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in targeted_ids) else 0
    )

    # order items by score
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    # trim to k
    ranking_topk = ranking.iloc[:k]

    # get labels
    y_true = [1 if r else 0 for r in ranking_topk["relevance"]]
    # convert "score" in ranking to decision threshold
    # higher score means higher ranked
    total_items = len(ranking_topk)
    y_score = [total_items - i for i in range(total_items)]
    # compute
    auc = float(roc_auc_score(y_true, y_score))
    return auc
