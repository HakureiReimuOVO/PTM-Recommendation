import numpy as np


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


# 计算DCG
def dcg(scores, k):
    return sum((2 ** scores[i] - 1) / np.log2(i + 2) for i in range(k))


# 计算NDCG
def ndcg(pred_scores, real_scores, k):
    # 根据pred_scores进行排序
    sorted_indices = np.argsort(pred_scores)[::-1][:k]
    sorted_real_scores = [real_scores[i] for i in sorted_indices]

    # 计算DCG
    actual_dcg = dcg(sorted_real_scores, k)

    # 计算理想情况下的DCG
    ideal_scores = sorted(real_scores, reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


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
def map_at_k(pred_scores, real_scores, k):
    sorted_indices = np.argsort(pred_scores)[::-1][:k]
    sorted_real_scores = [real_scores[i] for i in sorted_indices]

    num_relevant = 0
    sum_precisions = 0
    for i, score in enumerate(sorted_real_scores, start=1):
        if score > 0:
            num_relevant += 1
            sum_precisions += num_relevant / i

    return sum_precisions / num_relevant if num_relevant > 0 else 0


# 示例输入
# pred_scores = [0.6, 0.1, 0.4, 0.3, 0.3, 0.7]
pred_scores = [0.7, 0.6, 0.4, 0.9, 0.3, 0.5]
real_scores = [0.7, 0.6, 0.4, 0.9, 0.3, 0.5]
k = 4  # 推荐列表的长度

# 计算NDCG@k
ndcg_value = ndcg(pred_scores, real_scores, k)
print(f"NDCG@{k}: {ndcg_value:.4f}")

# 计算MRR@k
mrr_value = map_k(pred_scores, real_scores, k)
print(f"MRR@{k}: {mrr_value:.4f}")

# 计算MAP@k
map_value = map_k(pred_scores, real_scores, k)
print(f"MAP@{k}: {map_value:.4f}")

# 示例使用
arr = [50, 30, 50, 80, 70]
print(rank_array(arr))
