import torch
import os
import utils
import numpy as np
from tqdm import tqdm
from LogME.LogME import LogME 
from LogME.LEEP import LEEP
from LogME.NCE import NCE
from hf_dataset import get_hf_data_loader
from hf_model import get_hf_model_and_processor
from hf_localize import model_configs, dataset_configs


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

    print(f"Features shape: {features.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Targets shape: {targets.shape}")

    return features, outputs, targets


def scoring(model_name, dataset_name, model, data_loader):
    print(f'Calc Transferabilities of {model_name} on {dataset_name}')

    if torch.cuda.is_available():
        model = model.cuda()

    # Different models has different linear projection names
    try:
        if model_name in ['microsoft/resnet-18', 'microsoft/resnet-50']:
            fc_layer = model.classifier[-1]
        else:
            fc_layer = model.classifier
    except Exception:
        raise NotImplementedError

    print('Conducting features extraction...')
    features, outputs, targets = forward_pass(data_loader, model, fc_layer)

    print('Conducting transferability calculation...')
    logme = LogME(regression=False)
    score_logme = logme.fit(features.numpy(), targets.numpy())
    score_leep = LEEP(features.numpy(), targets.numpy())
    score_nce = NCE(np.argmax(features.numpy(), axis=1), targets.numpy())

    print(f'LogME of {model_name}: {score_logme}')
    print(f'LEEP of {model_name}: {score_leep}')
    print(f'NCE of {model_name}: {score_nce}')

    # Replace / with _ to avoid file path confusion
    model_name = model_name.replace('/', '_')
    dataset_name = dataset_name.replace('/', '_')

    res = utils.get_obj(f'result/{dataset_name}_score.json')
    res[model_name] = score_logme
    utils.save_obj(res, f'result/{dataset_name}_score.json')

    torch.save(logme.ms, f'result/{dataset_name}_{model_name}.pth')

    return score_logme


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Using cuda:')
        torch.device('cuda')
    else:
        print('Using cpu:')
        torch.device('cpu')

    if not os.path.isdir(f'result'):
        os.mkdir(f'result')

    for dataset in dataset_configs:
        score_data_loader = get_hf_data_loader(dataset_name=dataset['name'], image_key=dataset['image_key'],
                                               label_key=dataset['label_key'], batch_size=32, test=True)
        score_dict = {}
        for model in model_configs:
            score_model, _ = get_hf_model_and_processor(model_name=model)
            score_dict[model] = scoring(model_name=model, dataset_name=dataset['name'], model=score_model,
                                        data_loader=score_data_loader)

        utils.save_obj(score_dict, f"result/{dataset['name'].replace('/', '_')}_score_dict.json")
