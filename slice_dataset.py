from datasets import load_from_disk
from hf_config import dataset_configs, chunk_size


def load_sliced_dataset(dataset_name):
    dataset_config = dataset_configs[dataset_name]


if __name__ == '__main__':
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        dataset = load_from_disk(f"datasets/{dataset_name}")['train']

        # Ensure the way of slicing repeatable
        dataset = dataset.shuffle(seed=81)

        num_splits = (len(dataset) + chunk_size - 1) // chunk_size

        for i in range(num_splits):
            indices = range(i * chunk_size, min((i + 1) * chunk_size, len(dataset)))
            sub_dataset = dataset.select(indices)
            sub_dataset.save_to_disk(f"sliced_datasets/{dataset_name}_{i}")
