import numpy as np
import torch
from datasets import load_from_disk
from torchvision import models, transforms
from tqdm import tqdm

from config import dataset_configs


def _extract_features_by_resnet():
    """
    Extract features from all datasets
    Output all [dataset_name]_features.npy to directory 'features'
    """
    if torch.cuda.is_available():
        print('Using Cuda')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])

    if torch.cuda.is_available():
        model = model.cuda()

    # Extract the features of dataset
    for dataset_config in dataset_configs:
        img_features = []

        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        dataset = load_from_disk(f"datasets/{dataset_name}")['train']

        for item in tqdm(dataset):
            img = preprocess(item[image_key]).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()

            with torch.no_grad():
                img_feature = model(img).cpu().squeeze().numpy()

            img_features.append(img_feature)

        np.save(f"features/{dataset_name}_features.npy", np.array(img_features))


def get_dataset_features(dataset_name):
    """
    Get features from provided dataset
    Args:
        dataset_name
    """
    return np.load(f"features/{dataset_name}_features.npy")


if __name__ == '__main__':
    _extract_features_by_resnet()
    # features = get_dataset_features('cifar10')
    pass
