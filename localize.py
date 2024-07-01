import os
import config
from torch.utils.data import DataLoader
from model_loader import get_model_and_processor
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoImageProcessor, AutoModelForImageClassification
from dataset_loader import PreprocessedDataset, get_data_loader

model_configs = config.model_configs
dataset_configs = config.dataset_configs


def save_dataset(dataset_name):
    if not os.path.isdir(f"datasets/{dataset_name}"):
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(f"datasets/{dataset_name}")


def save_model_and_processor(model_name):
    if not os.path.isdir(f"models/{model_name}"):
        model = AutoModelForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.save_pretrained(f"models/{model_name}")
        processor.save_pretrained(f"models/{model_name}")


def save_to_local():
    """
    Save models and datasets from HuggingFace to local folder.

    :return: none
    """
    global model_configs, dataset_configs
    for model in model_configs:
        save_model_and_processor(model)
    for dataset in dataset_configs:
        save_dataset(dataset["name"])


def _test():
    """
    Test loading local datasets and models.

    :return: None
    """
    dataset_name = dataset_configs[0]["name"]
    model_name = model_configs[0]

    dataset = load_from_disk(f"datasets/{dataset_name}")["train"]
    model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}")
    processor = AutoImageProcessor.from_pretrained(f"models/{model_name}")

    score_dataset = PreprocessedDataset(
        data_list=dataset,
        image_key=dataset_configs[0]["image_key"],
        label_key=dataset_configs[0]["label_key"],
        processor=processor,
    )

    score_data_loader = DataLoader(score_dataset, batch_size=16, shuffle=False)


def _print_models():
    for model_config in model_configs:
        model, _ = get_model_and_processor(model_config)
        print(model)
        print("==========")


def _print_datasets():
    for dataset_config in dataset_configs:
        local_dataset = load_from_disk(f"datasets/{dataset_config['name']}")["train"]
        print(dataset_config["name"])
        print(local_dataset)
        print("==========")


if __name__ == "__main__":
    # model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18')
    # print(model)

    save_to_local()
    # _test()
    # _print_models()
    # _print_datasets()
    pass
