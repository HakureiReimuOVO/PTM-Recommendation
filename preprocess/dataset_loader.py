import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import root_path


class HFDataset(Dataset):
    def __init__(self, data_list, image_key, label_key, transform=None, processor=None):
        self.data = data_list
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][self.image_key]
        label = self.data[index][self.label_key]
        if self.transform:
            image = self.transform(image)
        elif self.processor:
            image = self.processor(image)
            image = torch.tensor(np.array(image['pixel_values'][0]))
        return image, label


def get_hf_data_loader(dataset_name,
                       image_key='image',
                       label_key='label',
                       batch_size=16,
                       image_size=224,
                       processor=None,
                       print_info=False,
                       test=False):
    dataset = load_from_disk(dataset_name)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if test:
        dataset = dataset.select(list(range(64)))
    if print_info:
        print('Dataset preview:')
        print(dataset)
        print('Data preview:')
        print(dataset[0])
    if processor:
        dataset = HFDataset(data_list=dataset, image_key=image_key, label_key=label_key, processor=processor)
    else:
        dataset = HFDataset(data_list=dataset, image_key=image_key, label_key=label_key, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if print_info:
        print('Data loaded successfully.')
    return data_loader
