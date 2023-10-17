"""
    Merged Torrent Sample:
    {
    'id': 'albert-base-v1',
    'LatestGitCommitSHA': 'aeffd769076a5c4f83b2546aea99ca45a15a5da4',
    'ModelHub': {'MetadataFilePath': 'data/huggingface/json/metadata/hf_metadata.json',
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

    Clustering/Embedding Attributes:
    1.ModelName
    2.ModelOwner
    3.ModelTask
    4.ModelArchitecture
    5.Dataset
"""

import numpy as np
import pickle
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from embedding import embed_attribute, embed_torrents

FILE_PATH = 'data/cleaned_metadata.json'
SAVE_PATH = 'data/torrent_embeddings.pkl'


def _cluster_torrents(torrents, method='KMeans', num_of_clusters=10):
    if method == 'KMeans':
        kmeans = KMeans(n_clusters=num_of_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(torrents)

        # Downscaling with PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(torrents)

        # Visualize
        colors = plt.cm.tab20(np.linspace(0, 1, num_of_clusters))
        plt.figure(figsize=(8, 6))
        for i in range(num_of_clusters):
            plt.scatter(embeddings_2d[cluster_labels == i, 0], embeddings_2d[cluster_labels == i, 1], color=colors[i])
        plt.title('Clustering of Torrents')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    elif method == 'DBscan':
        # dbscan = DBSCAN(eps=0.5, min_samples=5)
        # cluster_labels = dbscan.fit_predict(torrents)
        pass


def _sse_analysis(torrents, min_num_of_clusters=1, max_num_of_clusters=20):
    sse = []

    for k in range(min_num_of_clusters, max_num_of_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(torrents)
        sse.append(kmeans.inertia_)

    # Relationship between sse and num of clusters
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_num_of_clusters, max_num_of_clusters + 1), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.show()


def _cosine_similarity_indexing(embeddings, metadata, raw_input, k=5):
    input_embedding = embed_attribute(raw_input).reshape(1, -1)
    sim_scores = []

    for i in range(len(embeddings)):
        torrent_embedding = embeddings[i].reshape(4, 768)

        # Only index attribute 'ModelTask'
        # torrent_embedding = torrent_embedding[2].reshape(1, 768)

        cos_sim = cosine_similarity(torrent_embedding, input_embedding)
        sim_scores.append(np.sum(cos_sim))

    rank = np.argsort(sim_scores)[-k:][::-1]
    top_k = [sim_scores[i] for i in rank]

    for i in rank:
        print('Index result[%d]:' % i, metadata[i], )


if __name__ == '__main__':
    embed_torrents(file_path=FILE_PATH, save_path=SAVE_PATH)

    with open(SAVE_PATH, 'rb') as f:
        torrent_embeddings = pickle.load(f)

    with open(FILE_PATH, 'r') as f:
        torrent_metadata = json.load(f)

    # Select the best num of clusters
    _sse_analysis(torrents=torrent_embeddings, min_num_of_clusters=1, max_num_of_clusters=15)

    # Use K-Means to cluster
    _cluster_torrents(torrents=torrent_embeddings, method='KMeans', num_of_clusters=7)

    # TODO
    usr_input = input('Input text to indexing torrent\n>>')
    _cosine_similarity_indexing(embeddings=torrent_embeddings, metadata=torrent_metadata, raw_input=usr_input)
