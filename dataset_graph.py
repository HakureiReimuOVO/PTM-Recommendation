import json

import numpy as np
import torch

from tqdm import tqdm
from datasets import load_from_disk
from hf_config import *
from torchvision import models, transforms

DATASET_FEATURE_PATH = 'result/dataset_features.json'


def get_dataset_feature(dataset_name, features):
    return next((feature for feature in features if feature["dataset_name"] == dataset_name), None)


if __name__ == '__main__':
    with open(DATASET_FEATURE_PATH, 'r') as f:
        dataset_feature = json.load(f)

    # Initialize the model and preprocessor
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(pretrained=True)
    models.alexnet(pretrained=True)
    models.googlenet()
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])

    if torch.cuda.is_available():
        model = model.cuda()

    # Calculate the vector of datasets
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']
        unique_labels = get_dataset_feature(dataset_name, dataset_feature)['unique_labels']

        # 1. Slice
        num_splits = (dataset_config['num_rows'] + chunk_size - 1) // chunk_size
        for i in range(num_splits):
            dataset = load_from_disk(f"sliced_datasets/{dataset_name}_{i}")
            label_dict = dataset.features[label_key].names

            for specific_label in unique_labels:
                print(f'Processing {dataset_name}_{i}/{num_splits}: {specific_label}')

                selected_images = [item for item in dataset if label_dict[item[label_key]] == specific_label]

                # Optional: Reduce the amount of images
                selected_images = selected_images[:3]

                features = []
                for item in tqdm(selected_images):
                    img = preprocess(item[image_key]).unsqueeze(0)

                    if torch.cuda.is_available():
                        img = img.cuda()

                    with torch.no_grad():
                        feature = model(img)
                    features.append(feature.cpu().squeeze().numpy())
                avg_feature_vector = np.mean(features, axis=0)
                print(avg_feature_vector)

        # 2. No slice
        # dataset = load_from_disk(f"datasets/{dataset_name}")['train']
        # label_dict = dataset.features[label_key].names
        # for specific_label in unique_labels:
        #     print(f'Processing {dataset_name}: {specific_label}')
        #     label_dict = dataset.features[label_key].names
        #     selected_images = [item for item in dataset if label_dict[item[label_key]] == specific_label]
        #
        #     # Optional: Reduce the amount of images
        #     selected_images = selected_images[:10]
        #
        #     features = []
        #     for item in tqdm(selected_images):
        #         img = preprocess(item[image_key]).unsqueeze(0)
        #
        #         if torch.cuda.is_available():
        #             img = img.cuda()
        #
        #         with torch.no_grad():
        #             feature = model(img)
        #         features.append(feature.cpu().squeeze().numpy())
        #     avg_feature_vector = np.mean(features, axis=0)
