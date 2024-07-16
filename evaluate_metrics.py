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


def rmv(predicted_scores, ground_truth_scores):
    top1_index = np.argmax(predicted_scores)
    top1_ground_truth_score = ground_truth_scores[top1_index]
    max_ground_truth_score = np.max(ground_truth_scores)
    ratio = top1_ground_truth_score / max_ground_truth_score if max_ground_truth_score > 0 else 0
    if ifPrint:
        print("RMV:", ratio)
    return ratio


# 示例数据
# S = ['A', 'B', 'C', 'D', 'E']
# predicted_scores = [0.8, 0.3, 0.5, 0.9, 0.4]
# ground_truth_scores = [0.9, 0.1, 0.4, 0.8, 0.5]
# k = 3
#
# # 计算各个指标
# precision = precision_at_k(predicted_scores, ground_truth_scores, k)
# recall = recall_at_k(predicted_scores, ground_truth_scores, k)
# mrr = mrr_at_k(predicted_scores, ground_truth_scores, k)
# map_score = map_at_k(predicted_scores, ground_truth_scores, k)
# ndcg = ndcg_at_k(predicted_scores, ground_truth_scores, k)
#
# # 输出结果
# print("Precision@{}:".format(k), precision)
# print("Recall@{}:".format(k), recall)
# print("MRR@{}:".format(k), mrr)
# print("MAP@{}:".format(k), map_score)
# print("NDCG@{}:".format(k), ndcg)

#
#
def rank_array(arr):
    # 生成一个带有索引的数组
    indexed_arr = list(enumerate(arr))

    # 按照值降序排序
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)

    # 分配排名
    rank = 1
    ranks = [0] * len(arr)
    for i in range(len(sorted_arr)):
        if i > 0 and sorted_arr[i][1] < sorted_arr[i - 1][1]:
            rank = i + 1
        ranks[sorted_arr[i][0]] = rank

    return ranks


#
#
# # 计算DCG
# def dcg(scores, k):
#     return sum((2 ** scores[i] - 1) / np.log2(i + 2) for i in range(k))
#
#
# # 计算NDCG
# def ndcg(pred_scores, real_scores, k):
#     # 根据pred_scores进行排序
#     sorted_indices = np.argsort(pred_scores)[::-1][:k]
#     sorted_real_scores = [real_scores[i] for i in sorted_indices]
#
#     # 计算DCG
#     actual_dcg = dcg(sorted_real_scores, k)
#
#     # 计算理想情况下的DCG
#     ideal_scores = sorted(real_scores, reverse=True)[:k]
#     ideal_dcg = dcg(ideal_scores, k)
#
#     return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


# MRR@k
def map_k(pred_scores, real_scores, k):
    sorted_indices = np.argsort(pred_scores)[::-1][:k]
    real_scores_rank = rank_array(real_scores)
    ranks = sorted([real_scores_rank[idx] for idx in sorted_indices])
    res = 0
    count = 0
    for i, rank in enumerate(ranks):
        res += (i + 1) / rank
        count += 1
    return res / count

# 计算MAP@k
# def map_at_k(pred_scores, real_scores, k):
#     sorted_indices = np.argsort(pred_scores)[::-1][:k]
#     sorted_real_scores = [real_scores[i] for i in sorted_indices]
#
#     num_relevant = 0
#     sum_precisions = 0
#     for i, score in enumerate(sorted_real_scores, start=1):
#         if score > 0:
#             num_relevant += 1
#             sum_precisions += num_relevant / i
#
#     return sum_precisions / num_relevant if num_relevant > 0 else 0

#
# # 示例输入
# # pred_scores = [0.6, 0.1, 0.4, 0.3, 0.3, 0.7]
# pred_scores = [0.7, 0.6, 0.4, 0.9, 0.3, 0.5]
# real_scores = [0.7, 0.6, 0.4, 0.9, 0.3, 0.5]
# k = 4  # 推荐列表的长度
#
# # 计算NDCG@k
# ndcg_value = ndcg(pred_scores, real_scores, k)
# print(f"NDCG@{k}: {ndcg_value:.4f}")
#
# # 计算MRR@k
# mrr_value = map_k(pred_scores, real_scores, k)
# print(f"MRR@{k}: {mrr_value:.4f}")
#
# # 计算MAP@k
# map_value = map_k(pred_scores, real_scores, k)
# print(f"MAP@{k}: {map_value:.4f}")
#
# # 示例使用
# arr = [50, 30, 50, 80, 70]
# print(rank_array(arr))
