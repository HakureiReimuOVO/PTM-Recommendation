from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# dateset_names = ["cifar10", "cifar100", "beans", "snacks",
#                  "sasha/dog-food", "nelorth/oxford-flowers",
#                  "zh-plus/tiny-imagenet"]
# scoring_dataset = [load_dataset(dateset_name) for dateset_name in dateset_names]

# Configuration
BATCH_SIZE = 16


class MyDataset(Dataset):
    def __init__(self, data_list, image_key, label_key, transform=None):
        self.data = data_list
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][self.image_key]
        label = self.data[index][self.label_key]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_data_loader(dataset_name, image_key='image', label_key='label'):
    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train']
    print('========== Dataset Preview ==========')
    print(train_dataset)
    print('========== Data Preview ==========')
    print(train_dataset[0])

    train_dataset = MyDataset(data_list=train_dataset, image_key=image_key, label_key=label_key, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    print('========== Data Loaded Successfully ==========')
    return data_loader
