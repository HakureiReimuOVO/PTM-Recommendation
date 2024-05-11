import json

import numpy as np
from PIL.Image import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForImageClassification
from hf_config import dataset_configs

"""
sample = {'id': 'bert-base-uncased', 'LatestGitCommitSHA': '0a6aa9128b6194f4f3c4db429b6cb4891cdb421b',
     'ModelHub': {'MetadataFilePath': 'data/huggingface/json/metadata/hf_metadata.json',
                  'MetadataObjectID': 'bert-base-uncased', 'ModelHubName': 'Hugging Face',
                  'ModelHubURL': 'https://huggingface.co/'}, 'ModelName': 'bert-base-uncased',
     'ModelOwner': 'http://huggingface.co/', 'ModelOwnerURL': 'http://huggingface.co/',
     'ModelURL': 'https://huggingface.co/bert-base-uncased', 'ModelArchitecture': 'bert', 'ModelTask': 'fill-mask',
     'modelId': 'bert-base-uncased', 'sha': '0a6aa9128b6194f4f3c4db429b6cb4891cdb421b',
     'lastModified': '2022-11-16T15:15:39.000Z',
     'tags': ['pytorch', 'tf', 'jax', 'rust', 'safetensors', 'bert', 'fill-mask', 'en', 'dataset:bookcorpus',
              'dataset:wikipedia', 'arxiv:1810.04805', 'transformers', 'exbert', 'license:apache-2.0',
              'autotrain_compatible', 'has_space'], 'pipeline_tag': 'fill-mask',
     'siblings': [{'rfilename': '.gitattributes', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'LICENSE', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'README.md', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'config.json', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'flax_model.msgpack', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'model.safetensors', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'pytorch_model.bin', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'rust_model.ot', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'tf_model.h5', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'tokenizer.json', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'tokenizer_config.json', 'size': None, 'blob_id': None, 'lfs': None},
                  {'rfilename': 'vocab.txt', 'size': None, 'blob_id': None, 'lfs': None}], 'private': False,
     'author': None, 'config': {'architectures': ['BertForMaskedLM'], 'model_type': 'bert'}, 'securityStatus': None,
     '_id': '621ffdc036468d709f174338', 'cardData': {'language': 'en', 'tags': ['exbert'], 'license': 'apache-2.0',
                                                     'datasets': ['bookcorpus', 'wikipedia']}, 'likes': 464,
     'downloads': 24420435, 'library_name': 'transformers'}
"""

dataset_dump_path = 'result/dataset_features.json'
model_dump_path = 'result/model_features.json'
torrent_path = 'ptm_torrent/torrent_data/top_1000_downloads_metadata.json'
model_configs = ['apple/mobilevit-small',  # model.classifier
                 'facebook/convnextv2-tiny-1k-224',  # model.classifier
                 'facebook/convnextv2-tiny-22k-384',  # model.classifier
                 'google/mobilenet_v1_0.75_192',  # model.classifier
                 'google/mobilenet_v2_1.0_224',  # model.classifier
                 'google/vit-base-patch16-224',  # model.classifier
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
with open(torrent_path, 'r') as f:
    torrents = json.load(f)


# 抽取数据集特征
def extract_dataset_features(dataset_config):
    dataset_name = dataset_config['name']
    image_key = dataset_config['image_key']
    label_key = dataset_config['label_key']
    dataset = load_from_disk(f'datasets/{dataset_name}')['train']

    sample_image = dataset[0][image_key]
    if isinstance(sample_image, Image):
        image_size = sample_image.size
        num_channels = len(sample_image.getbands())
    elif isinstance(sample_image, np.ndarray):
        image_size = sample_image.shape[1], sample_image.shape[0]
        num_channels = sample_image.shape[2] if len(sample_image.shape) > 2 else 1
    else:
        raise TypeError("Unknown type for image data")

    num_samples = len(dataset)

    labels = dataset.features[label_key].names
    num_classes = len(labels)

    return {
        'dataset_name': dataset_name,
        'image_size': image_size,
        'num_channels': num_channels,

        'num_samples': num_samples,
        'num_classes': num_classes,
        'unique_labels': labels
    }


def extract_model_features(model_name):
    model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}")

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_layers = len(list(model.modules()))

    # TODO
    # print(model.config)
    input_size = model.config.image_size if hasattr(model.config, 'image_size') else (224, 224)
    num_classes = model.config.num_labels if hasattr(model.config, 'num_labels') else None

    match_torrent = {}
    for torrent in torrents:
        if torrent['id'] == model_name:
            match_torrent = torrent

    pretrained_dataset = "None"
    if 'tags' in match_torrent:
        for tag in match_torrent['tags']:
            prefix = "dataset:"
            if tag.startswith(prefix):
                pretrained_dataset = tag[len(prefix):]

    model_type = model.__class__.__name__

    # 返回抽取的特征
    return {
        'model_name': model_name,
        'pretrained_dataset': pretrained_dataset,
        'num_parameters': num_parameters,
        'num_layers': num_layers,
        'input_size': input_size,
        'num_classes': num_classes,
        'model_type': model_type,
        'model_owner': match_torrent['ModelOwner'] if 'ModelOwner' in match_torrent else 'None',
        'model_architecture': match_torrent['ModelArchitecture'] if 'ModelArchitecture' in match_torrent else 'None',
        'model_task': match_torrent['ModelTask'] if 'ModelTask' in match_torrent else 'None',
        'downloads': match_torrent['downloads'] if 'downloads' in match_torrent else 'None',
    }


# 对每个数据集配置抽取特征
all_dataset_features = []
for config in dataset_configs:
    features = extract_dataset_features(config)
    all_dataset_features.append(features)

for features in all_dataset_features:
    print(features)

# 对每个模型抽取特征
all_model_features = []
for model_name in model_configs:
    features = extract_model_features(model_name)
    all_model_features.append(features)

for features in all_model_features:
    print(features)

with open(dataset_dump_path, 'w') as f:
    json.dump(all_dataset_features, f)

with open(model_dump_path, 'w') as f:
    json.dump(all_model_features, f)
