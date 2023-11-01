import torch
import os
import tqdm
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from LogME.LogME import LogME

# Configuration
BATCH_SIZE = 48

dataset = 'dog_photos'
dataset_path = 'data/dog_photos'
score_models = ['resnet50']


# score_models = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']

def forward_pass(data_loader, model, fc_layer):
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(_, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    print("start")
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            targets.append(target)
            # data = data.cuda()
            _ = model(data)

    forward_hook.remove()
    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])

    print(features.shape, outputs.shape, targets.shape)

    return features, outputs, targets


def score(model_name, dataset_name, data_loader):
    print(f'Calc Transferabilities of {model_name} on {dataset}')

    model = models.__dict__[model_name](pretrained=True)
    # model = model.cuda()

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

    # save calculated bayesian weight
    if not os.path.isdir(f'logme_{dataset_name}'):
        os.mkdir(f'logme_{dataset_name}')

    torch.save(logme.ms, f'logme_{dataset_name}/weight_{model_name}.pth')

    print(f'LogME of {model_name}: {score}\n')
    return score


# torch.device('cuda')
torch.device('cpu')


# Data transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

score_dataset = datasets.ImageFolder(dataset_path, transform=transform)
score_loader = DataLoader(score_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

score_dict = {}
for model in score_models:
    score_dict[model] = score(model_name=model, dataset_name=dataset, data_loader=score_loader)
