import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
import torch
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


def embed_attribute(attribute):
    tokens = tokenizer(attribute, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)

    # Apply BERT's [CLS] header output as embedding
    embeddings = outputs.last_hidden_state[:, 0, :].numpy().squeeze()

    return embeddings


def _embed_torrent(torrent):
    name_vector = embed_attribute(torrent['ModelName'])
    owner_vector = embed_attribute(torrent['ModelOwner'])
    task_vector = embed_attribute(torrent['ModelTask'])
    architecture_vector = embed_attribute(torrent['ModelArchitecture'])

    # Get vector of shape (4*768) after concatenate
    torrent_vector = np.concatenate((name_vector, owner_vector, task_vector, architecture_vector))

    return torrent_vector


def embed_torrents(file_path: str, save_path: str):
    with open(file_path, 'r') as f:
        ptm_torrents = json.load(f)

    torrent_embeddings = []

    for i in range(len(ptm_torrents)):
        # TODO: progress visualization, multi-thread
        vector = _embed_torrent(ptm_torrents[i])
        torrent_embeddings.append(vector)

    with open(save_path, 'wb') as f:
        pickle.dump(torrent_embeddings, f)
