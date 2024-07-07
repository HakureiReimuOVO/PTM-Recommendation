import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class PreprocessedDataset(Dataset):
    def __init__(self, data_list, image_key, label_key, transform=None, processor=None, label_encoder=None):
        self.data = data_list
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.processor = processor
        self.label_encoder = label_encoder

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
        if self.label_encoder:
            label = self.label_encoder.transform([label])[0]
        return image, label


def get_data_loader(dataset_name,
                    image_key='image',
                    label_key='label',
                    batch_size=16,
                    image_size=224,
                    processor=None,
                    print_info=False,
                    test=False,
                    train_test_split_ratio=0.8):
    dataset = load_from_disk(f'preprocessed_datasets_test/{dataset_name}')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_encoder = LabelEncoder()
    labels = [example[label_key] for example in dataset]
    label_encoder.fit(labels)

    if test:
        dataset = dataset.select(list(range(100)))
    if print_info:
        print('Dataset preview:')
        print(dataset)
        print('Data preview:')
        print(dataset[0])
    if processor:
        dataset = PreprocessedDataset(data_list=dataset, image_key=image_key, label_key=label_key, processor=processor,
                                      label_encoder=label_encoder)
    else:
        dataset = PreprocessedDataset(data_list=dataset, image_key=image_key, label_key=label_key, transform=transform,
                                      label_encoder=label_encoder)

    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if print_info:
        print('Data loaded successfully.')
    return train_loader, test_loader
