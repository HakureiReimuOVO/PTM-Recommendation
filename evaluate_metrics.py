import numpy as np

ifPrint = False


# Precision@K
def precision_at_k(predicted_scores, ground_truth_scores, k):
    recommended_indices = np.argsort(predicted_scores)[::-1][:k]
    relevant_indices = np.argsort(ground_truth_scores)[::-1][:k]
    relevant_count = sum([1 for idx in recommended_indices if idx in relevant_indices])
    precision = relevant_count / k
    if ifPrint:
        print("Precision@{}:".format(k), precision)
    return precision


# Recall@K
def recall_at_k(predicted_scores, ground_truth_scores, k):
    recommended_indices = np.argsort(predicted_scores)[::-1][:k]
    relevant_indices = np.argsort(ground_truth_scores)[::-1][:k]
    relevant_count = sum([1 for idx in recommended_indices if idx in relevant_indices])
    recall = relevant_count / len(relevant_indices)
    if ifPrint:
        print("Recall@{}:".format(k), recall)
    return recall


# MRR@K
def mrr_at_k(predicted_scores, ground_truth_scores, k):
    recommended_indices = np.argsort(predicted_scores)[::-1][:k]
    relevant_indices = np.argsort(ground_truth_scores)[::-1][:k]
    for rank, idx in enumerate(recommended_indices):
        if idx in relevant_indices:
            mrr = 1.0 / (rank + 1)
            if ifPrint:
                print("MRR@{}:".format(k), mrr)
            return mrr
    if ifPrint:
        print("MRR@{}:".format(k), 0)
    return 0


# MAP@K
def map_at_k(predicted_scores, ground_truth_scores, k):
    recommended_indices = np.argsort(predicted_scores)[::-1][:k]
    relevant_indices = np.argsort(ground_truth_scores)[::-1][:k]
    relevant_count = 0
    precision_scores = []

    for i in range(min(k, len(recommended_indices))):
        if recommended_indices[i] in relevant_indices:
            relevant_count += 1
            precision_scores.append(relevant_count / (i + 1))

    if not precision_scores:
        if ifPrint:
            print("MAP@{}:".format(k), 0.0)
        return 0.0
    map_score = sum(precision_scores) / min(k, len(relevant_indices))
    if ifPrint:
        print("MAP@{}:".format(k), map_score)
    return map_score


# NDCG@K
def dcg_at_k(scores, k):
    return sum((2 ** scores[i] - 1) / np.log2(i + 2) for i in range(k))


def ndcg_at_k(predicted_scores, ground_truth_scores, k):
    recommended_indices = np.argsort(predicted_scores)[::-1][:k]
    sorted_real_scores = [ground_truth_scores[i] for i in recommended_indices]

    actual_dcg = dcg_at_k(sorted_real_scores, k)
    ideal_scores = sorted(ground_truth_scores, reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_scores, k)

    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    if ifPrint:
        print("NDCG@{}:".format(k), ndcg)
    return ndcg


def rmv(predicted_scores, ground_truth_scores, k=1):
    top_k_predicted_indices = np.argsort(predicted_scores)[::-1][:k]
    top_k_ground_truth_indices = np.argsort(ground_truth_scores)[::-1][:k]

    rmv_values = []
    for pred_idx, gt_idx in zip(top_k_predicted_indices, top_k_ground_truth_indices):
        pred_score = ground_truth_scores[pred_idx]
        gt_score = ground_truth_scores[gt_idx]
        rmv_value = pred_score / gt_score if gt_score > 0 else 0
        rmv_values.append(rmv_value)

    avg_rmv = np.mean(rmv_values)

    if ifPrint:
        print("RMV@{}:".format(k), avg_rmv)

    return avg_rmv
