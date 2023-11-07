from torch.utils.data import DataLoader
from hf_dataset import HFDataset, get_hf_data_loader
from hf_model import get_hf_model_and_processor
from datasets import load_dataset, load_from_disk
from transformers import (AutoModel,
                          AutoImageProcessor,
                          AutoModelForImageClassification)

model_configs = [
    'microsoft/resnet-18',  # model.classifier[-1]
    'microsoft/resnet-50',  # model.classifier[-1]
    'nateraw/vit-age-classifier',  # model.classifier
    'google/vit-base-patch16-224',  # model.classifier
    'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
]

dataset_configs = [
    {
        'name': 'cifar10',
        'image_key': 'img',
        'label_key': 'label'
    },
    {
        'name': 'cifar100',
        'image_key': 'img',
        'label_key': 'fine_label'
    },
    {
        'name': 'beans',
        'image_key': 'image',
        'label_key': 'labels'
    },
    {
        'name': 'Matthijs/snacks',
        'image_key': 'image',
        'label_key': 'label'
    },
    {
        'name': 'sasha/dog-food',
        'image_key': 'image',
        'label_key': 'label'
    },
]


def save_dataset(dataset_name):
    # if not os.path.isdir(f'datasets/{dataset_name}'):
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(f'datasets/{dataset_name}')


def save_model_and_processor(model_name):
    # if not os.path.isdir(f'models/{model_name}'):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.save_pretrained(f'models/{model_name}')
    processor.save_pretrained(f'models/{model_name}')


def save_to_local():
    """
    Save models and datasets from HuggingFace to local folder.

    :return: none
    """
    global model_configs, dataset_configs
    for model in model_configs:
        save_model_and_processor(model)
    for dataset in dataset_configs:
        save_dataset(dataset['name'])


def _test():
    """
    Test loading local datasets and models.

    :return: None
    """
    dataset_name = dataset_configs[0]['name']
    model_name = model_configs[0]

    dataset = load_from_disk(f"datasets/{dataset_name}")['train']
    model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}")
    processor = AutoImageProcessor.from_pretrained(f"models/{model_name}")

    score_dataset = HFDataset(data_list=dataset, image_key=dataset_configs[0]['image_key'],
                              label_key=dataset_configs[0]['label_key'],
                              processor=processor)

    score_data_loader = DataLoader(score_dataset, batch_size=16, shuffle=False)


if __name__ == '__main__':
    # save_to_local()
    # _test()
    for model_config in model_configs:
        model, _ = get_hf_model_and_processor(model_config)
        print(model)
    pass
