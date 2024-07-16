import os
import json
from itertools import combinations
from datasets import load_from_disk

from feature_extraction import get_dataset_features
from config import dataset_configs, chunk_size, root_path


# Deprecated
# def _slice_dataset_by_size():
#     for dataset_config in dataset_configs:
#         dataset_name = dataset_config["name"]
#         dataset = load_from_disk(f"datasets/{dataset_name}")["train"]
#
#         # Ensure the way of slicing repeatable
#         dataset = dataset.shuffle(seed=81)
#
#         # num_splits = (len(dataset) + chunk_size - 1) // chunk_size
#
#         num_splits = (dataset_config["num_rows"] + chunk_size - 1) // chunk_size
#
#         for i in range(num_splits):
#             indices = range(i * chunk_size, min((i + 1) * chunk_size, len(dataset)))
#             sub_dataset = dataset.select(indices)
#             sub_dataset.save_to_disk(f"{root_path}/{dataset_name}_{i}")


def _slice_dataset_by_label():
    """
    Slice the dataset by all combinations of k labels
    Output sliced datasets and index_map to directory [root_path]
    """
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        label_key = dataset_config['label_key']
        for k in dataset_config['comb']:
            dataset = load_from_disk(f"datasets/{dataset_name}")['train']
            # Test
            dataset = dataset.select(range(10000))
            dataset = dataset.shuffle(seed=81)
            label_dict = dataset.features[label_key].names
            label_indices = {label: [] for label in label_dict}

            for idx, example in enumerate(dataset):
                example_label = example[label_key]
                label_indices[label_dict[example_label]].append(idx)

            label_combinations = list(combinations(label_dict, k))

            for comb in label_combinations:
                combined_indices = set()
                for label in comb:
                    combined_indices.update(label_indices[label])

                combined_indices = sorted(list(combined_indices))

                if combined_indices:
                    sub_dataset = dataset.select(combined_indices)
                    comb_name = "_".join(map(str, comb))
                    sub_dataset_path = f"{root_path}/{dataset_name}_{comb_name}"
                    sub_dataset.save_to_disk(sub_dataset_path)

                    index_map_path = os.path.join(sub_dataset_path, "index_map.json")
                    with open(index_map_path, 'w') as f:
                        json.dump(list(combined_indices), f)


def get_all_datasets_and_idx(dataset_name):
    """
    Get all datasets, index_map, new_name by dataset nane
    Args:
        dataset_name
    """
    datasets = []

    for dataset_config in dataset_configs:
        if dataset_name == dataset_config['name']:
            for k in dataset_config['comb']:
                label_key = dataset_config['label_key']
                dataset = load_from_disk(f"datasets/{dataset_name}")['train']
                label_dict = dataset.features[label_key].names
                label_combinations = list(combinations(label_dict, k))

                for comb in label_combinations:
                    comb_name = "_".join(map(str, comb))
                    sub_dataset_path = f"{root_path}/{dataset_name}_{comb_name}"
                    sub_dataset = load_from_disk(sub_dataset_path)

                    index_map_path = os.path.join(sub_dataset_path, "index_map.json")
                    with open(index_map_path, 'r') as f:
                        index_map = json.load(f)

                    datasets.append((sub_dataset, index_map, f"{dataset_name}_{comb_name}"))

    return datasets


if __name__ == "__main__":
    _slice_dataset_by_label()
    # dataset = load_from_disk(f"../preprocessed_datasets_test/cifar10_cat_dog")
    # for idx, example in enumerate(dataset):
    #     print(example)

    # items = get_all_datasets_and_idx(dataset_name='cifar10')
    # print(items)
    # features = get_dataset_features(dataset_name='cifar10')
    #
    # for dataset, index_map in items[:1]:
    #     for idx, img in enumerate(dataset):
    #         new_idx = index_map[idx]
    #         print(idx, new_idx)
    #         feature = features[new_idx]
    #         print(img)
    pass
