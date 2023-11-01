"""
    Merged Torrent Sample:
    {
    'id': 'albert-base-v1',
    'LatestGitCommitSHA': 'aeffd769076a5c4f83b2546aea99ca45a15a5da4',
    'ModelHub': {'MetadataFilePath': 'torrent_data/huggingface/json/metadata/hf_metadata.json',
                 'MetadataObjectID': 'albert-base-v1', 'ModelHubName': 'Hugging Face',
                 'ModelHubURL': 'https://huggingface.co/'},
    'ModelName': 'albert-base-v1',
    'ModelOwner': 'http://huggingface.co/',
    'ModelOwnerURL': 'http://huggingface.co/',
    'ModelURL': 'https://huggingface.co/albert-base-v1',
    'ModelArchitecture': 'albert',
    'ModelTask': 'fill-mask',
    'modelId': 'albert-base-v1',
    'sha': 'aeffd769076a5c4f83b2546aea99ca45a15a5da4',
    'lastModified': '2021-01-13T15:08:24.000Z',
    'tags': ['pytorch', 'tf', 'albert', 'fill-mask', 'en', 'dataset:bookcorpus', 'dataset:wikipedia',
             'arxiv:1909.11942', 'transformers', 'exbert', 'license:apache-2.0', 'autotrain_compatible', 'has_space'],
    'pipeline_tag': 'fill-mask',
    'siblings': [{'rfilename': '.gitattributes', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'README.md', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'config.json', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'pytorch_model.bin', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'spiece.model', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'tf_model.h5', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'tokenizer.json', 'size': None, 'blob_id': None, 'lfs': None},
                 {'rfilename': 'with-prefix-tf_model.h5', 'size': None, 'blob_id': None, 'lfs': None}],
    'private': False,
    'author': None,
    'config': {'architectures': ['AlbertForMaskedLM'], 'model_type': 'albert'},
    'securityStatus': None,
    '_id': '621ffdc036468d709f174328',
    'cardData': {'tags': ['exbert'], 'language': 'en', 'license': 'apache-2.0', 'datasets': ['bookcorpus', 'wikipedia']},
    'likes': 1,
    'downloads': 73054,
    'library_name': 'transformers'
    }
"""
import json

DATA_PATH = '../torrent_data/cleaned_metadata.json'


def get_top_k_downloads_models(save_path: str, k=1000):
    with open(DATA_PATH, 'r') as f:
        metadata = json.load(f)
    for item in metadata:
        if 'downloads' not in item:
            item['downloads'] = 0
    sorted_metadata = sorted(metadata, key=lambda x: x['downloads'], reverse=True)
    top_k_metadata = sorted_metadata[:k]
    with open(save_path, 'w') as f:
        json.dump(top_k_metadata, f)


def print_model_downloads_count(data_path: str):
    with open(data_path, 'r') as f:
        metadata = json.load(f)
    for i in range(len(metadata)):
        item = metadata[i]
        print(f"{i}: {item['ModelName']} Downloads:{item['downloads']}")


if __name__ == '__main__':
    # get_top_k_downloads_models('../torrent_data/top_1000_downloads_metadata.json', 1000)
    print_model_downloads_count('../torrent_data/top_1000_downloads_metadata.json')
