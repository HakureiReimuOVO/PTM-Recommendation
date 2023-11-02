import torch
import os
from tqdm import tqdm
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from LogME.LogME import LogME
from dataset import get_data_loader

# Configuration
BATCH_SIZE = 16

# Get dataset from folder
dataset = 'dog_photos'
dataset_path = 'data/dog_photos_small'

# Get datasets from HuggingFace lib
dataset = [
    {
        'name': 'mnist',
        'image_key': 'image',
        'label_key': 'label'
    },
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
        'label_key': 'labels'
    },
    {
        'name': 'sasha/dog-food',
        'image_key': 'image',
        'label_key': 'label'
    },
]

# Get models from TorchVision lib
score_models = ['resnet50', 'resnet101', 'densenet169', 'googlenet']
# score_models = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']

def forward_pass(data_loader, model, fc_layer):
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(_, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    print("Start:")
    model.eval()
    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            targets.append(target)
            if torch.cuda.is_available():
                data = data.cuda()
            _ = model(data)
    forward_hook.remove()
    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])

    print(f"Features: {features.shape}")
    print(f"Outputs: {outputs.shape}")
    print(f"Targets: {targets.shape}")

    return features, outputs, targets


def score(model_name, dataset_name, data_loader):
    print(f'Calc Transferabilities of {model_name} on {dataset}')

    model = models.__dict__[model_name](pretrained=True)

    if torch.cuda.is_available():
        model = model.cuda()


    print(model)

    # different models has different linear projection names
    if model_name in ['mobilenet_v2', 'mnasnet1_0']:
        fc_layer = model.classifier[-1]
    elif model_name in ['densenet121', 'densenet169', 'densenet201']:
        fc_layer = model.classifier
    elif model_name in ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']:
        fc_layer = model.fc
    else:
        raise NotImplementedError

    print('Conducting features extraction...')
    features, outputs, targets = forward_pass(data_loader, model, fc_layer)
    # predictions = F.softmax(outputs)

    print('Conducting transferability calculation...')
    logme = LogME(regression=False)
    score = logme.fit(features.numpy(), targets.numpy())

    print(f'LogME of {model_name}: {score}\n')

    # save calculated bayesian weight
    if not os.path.isdir(f'logme_{dataset_name}'):
        os.mkdir(f'logme_{dataset_name}')

    torch.save(logme.ms, f'logme_{dataset_name}/weight_{model_name}.pth')
    return score

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')

    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # score_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    # score_loader = DataLoader(score_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    data = dataset[0]
    score_loader = get_data_loader(dataset_name=data['name'], image_key=data['image_key'], label_key=data['label_key'])

    score_dict = {}
    for model in score_models:
        score_dict[model] = score(model_name=model, dataset_name=data['name'], data_loader=score_loader)
