import os
from torch.utils.data import DataLoader
from hf_dataset import HFDataset, get_hf_data_loader
from hf_model import get_hf_model_and_processor
from datasets import load_dataset, load_from_disk
import transformers
from transformers import (AutoModel,
                          AutoImageProcessor,
                          AutoModelForImageClassification)

# model_configs = [
#     # 'microsoft/resnet-18',  # model.classifier[-1]
#     # 'microsoft/resnet-50',  # model.classifier[-1]
#     # 'nateraw/vit-age-classifier',  # model.classifier
#     # 'google/vit-base-patch16-224',  # model.classifier
#     # 'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
#     'facebook/convnext-tiny-224',  # model.classifier
#     # 'microsoft/swin-tiny-patch4-window7-224'
# ]

model_configs = ['apple/mobilevit-small',  # model.classifier
                 'facebook/convnextv2-tiny-1k-224',  # model.classifier
                 'facebook/convnextv2-tiny-22k-384',  # model.classifier
                 'google/mobilenet_v1_0.75_192',  # model.classifier
                 'google/mobilenet_v2_1.0_224',  # model.classifier
                 'google/vit-base-patch16-224',  # model.classifier
                 # 'google/vit-base-patch16-384',  # model.classifier
                 # 'google/vit-large-patch32-384',  # model.classifier
                 'microsoft/beit-base-patch16-224',  # model.classifier
                 'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
                 'microsoft/dit-base-finetuned-rvlcdip',  # model.classifier
                 'microsoft/resnet-18',  # model.classifier[-1]
                 'microsoft/resnet-50',  # model.classifier[-1]
                 'microsoft/swin-base-patch4-window7-224-in22k',  # model.classifier
                 'microsoft/swin-tiny-patch4-window7-224',  # model.classifier
                 'nateraw/vit-age-classifier',  # model.classifier
                 'nvidia/mit-b0',  # mode3l.classifier
                 'nvidia/mit-b2'  # model.classifier
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
    {
        'name': 'nelorth/oxford-flowers',
        'image_key': 'image',
        'label_key': 'label'
    },
    {
        'name': 'cats_vs_dogs',
        'image_key': 'image',
        'label_key': 'labels'
    }
]


def save_dataset(dataset_name):
    if not os.path.isdir(f'datasets/{dataset_name}'):
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(f'datasets/{dataset_name}')


def save_model_and_processor(model_name):
    if not os.path.isdir(f'models/{model_name}'):
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


def _print_models():
    for model_config in model_configs:
        model, _ = get_hf_model_and_processor(model_config)
        print(model)
        print('==========')


def _print_datasets():
    for dataset_config in dataset_configs:
        local_dataset = load_from_disk(f"datasets/{dataset_config['name']}")['train']
        print(dataset_config['name'])
        print(local_dataset)
        print('==========')


if __name__ == '__main__':
    # model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18')
    # print(model)
    # save_to_local()
    # _test()
    # _print_models()
    # dataset = load_dataset(f"datasets/sasha")['train']
    _print_datasets()
    # print(dataset)
    pass
