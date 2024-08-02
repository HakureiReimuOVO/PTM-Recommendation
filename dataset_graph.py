import json
import pickle
import time

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datasets import load_from_disk
from config import *
from torchvision import models, transforms

from feature_extraction import get_dataset_features
from slice_dataset import get_all_datasets_and_idx

FEATURE_OUTPUT_PATH = 'result/label_features.npy'
DATASET_FEATURE_OUTPUT_PATH = 'result/dataset_features.npy'
RATIO_OUTPUT_PATH = 'result/label_ratio.json'
NUMERIC_FEATURE_OUTPUT_PATH = 'result/dataset_numeric_features.json'
DATASET_GRAPH_OUTPUT_PATH = 'result/dataset_graph_mean.pkl'


def get_dataset_feature(dataset_name, features):
    return next((feature for feature in features if feature["dataset_name"] == dataset_name), None)


def get_dataset_mean_features():
    dataset_features = {}

    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name=dataset_name)
        features = get_dataset_features(dataset_name=dataset_name)

        for dataset, index_map, full_name in tqdm(items):
            label_dict = dataset.features[label_key].names

            total_features = []
            for idx, item in enumerate(dataset):
                label = label_dict[item[label_key]]
                new_idx = index_map[idx]
                feature = features[new_idx]
                total_features.append(feature)

            dataset_features[full_name] = np.mean(total_features, axis=0)

    np.save(DATASET_FEATURE_OUTPUT_PATH, dataset_features)

    # with open(FEATURE_OUTPUT_PATH, 'w') as f:
    #     json.dump(avg_label_features, f)

    return dataset_features


def get_label_features():
    label_features = {}

    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name=dataset_name)
        features = get_dataset_features(dataset_name=dataset_name)

        for dataset, index_map, _ in tqdm(items):
            label_dict = dataset.features[label_key].names

            for idx, item in enumerate(dataset):
                label = label_dict[item[label_key]]
                new_idx = index_map[idx]
                feature = features[new_idx]

                if label not in label_features:
                    label_features[label] = []
                label_features[label].append(feature)

    avg_label_features = {}
    for label, features in label_features.items():
        avg_feature_vector = np.mean(features, axis=0)
        print(f'Label: {label}, Average Feature Vector: {avg_feature_vector}')
        avg_label_features[label] = avg_feature_vector

    np.save(FEATURE_OUTPUT_PATH, avg_label_features)

    # with open(FEATURE_OUTPUT_PATH, 'w') as f:
    #     json.dump(avg_label_features, f)

    return avg_label_features


def get_label_ratios():
    dataset_ratios = {}
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name=dataset_name)

        for dataset, _, dataset_fullname in tqdm(items):

            label_dict = dataset.features[label_key].names
            label_counts = {}

            for item in dataset:
                label = label_dict[item[label_key]]
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            total_labels = sum(label_counts.values())
            label_ratios = {label: count / total_labels for label, count in label_counts.items()}
            dataset_ratios[dataset_fullname] = label_ratios

    with open(RATIO_OUTPUT_PATH, 'w') as f:
        json.dump(dataset_ratios, f)

    return dataset_ratios


def get_numeric_features():
    numeric_features = {}
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name=dataset_name)
        for dataset, _, dataset_fullname in tqdm(items):
            num_images = len(dataset)
            image_sizes = [item[image_key].size for item in dataset]
            avg_image_size = (
                    sum(size[0] * size[1] for size in image_sizes) // len(image_sizes)
            ) if image_sizes else 0
            num_classes = len(set(item[label_key] for item in dataset))

            # num_images = dataset_fullname.count('_') * 5000
            # avg_image_size = 1024
            # num_classes = dataset_fullname.count('_')

            numeric_features[dataset_fullname] = {
                'num_images': num_images,
                'image_size': avg_image_size,
                'num_classes': num_classes
            }

    with open(NUMERIC_FEATURE_OUTPUT_PATH, 'w') as f:
        json.dump(numeric_features, f)
    return numeric_features


def load_data(feature_path, ratio_path, numeric_feature_path):
    label_features = np.load(feature_path, allow_pickle=True).item()

    with open(ratio_path, 'r') as f:
        dataset_ratios = json.load(f)

    with open(numeric_feature_path, 'r') as f:
        numeric_features = json.load(f)

    # with open(feature_path, 'r') as f:
    #     label_features = json.load(f)

    return label_features, dataset_ratios, numeric_features


def create_graph(label_features, dataset_ratios, numeric_features):
    G = nx.DiGraph()

    numeric_features_dim = 256
    num_feats = np.array(
        [[data["num_images"], data["image_size"], data["num_classes"]] for data in numeric_features.values()])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(num_feats)
    expanded_features = np.repeat(normalized_features, numeric_features_dim, axis=1)
    num_feat_dict = {name: expanded_features[i] for i, name in enumerate(numeric_features.keys())}

    for label, features in label_features.items():
        # label_feat = np.array(features)
        # norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        # G.add_node(label, type='label', feature=label_feat / norm)

        label_feat = np.concatenate((np.array(features), np.zeros(3 * numeric_features_dim)))
        norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        G.add_node(label, type='label', feature=label_feat / norm)

    for dataset_name, labels_info in dataset_ratios.items():
        G.add_node(dataset_name, type='dataset')
        total_feature = np.zeros(len(next(iter(label_features.values()))))
        total_weight = 0

        for label, ratio in labels_info.items():
            if label in label_features:
                weight = ratio
                feature = np.array(label_features[label])
                weighted_feature = weight * feature
                total_feature += weighted_feature
                total_weight += weight
                G.add_edge(dataset_name, label, weight=weight)

        if total_weight > 0:
            # Add norm
            feature = total_feature / total_weight
            num_feat = num_feat_dict[dataset_name]
            cat_feat = np.concatenate((feature, num_feat))
            norm = np.linalg.norm(cat_feat, axis=-1, keepdims=True)
            G.nodes[dataset_name]['feature'] = cat_feat / norm
        else:
            G.nodes[dataset_name]['feature'] = np.zeros(len(total_feature) + 3 * numeric_features_dim)

    return G

def create_numeric_graph(label_features, dataset_ratios, numeric_features):
    G = nx.DiGraph()

    numeric_features_dim = 256
    num_feats = np.array(
        [[data["num_images"], data["image_size"], data["num_classes"]] for data in numeric_features.values()])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(num_feats)
    expanded_features = np.repeat(normalized_features, numeric_features_dim, axis=1)
    num_feat_dict = {name: expanded_features[i] for i, name in enumerate(numeric_features.keys())}

    for label, features in label_features.items():
        # label_feat = np.array(features)
        # norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        # G.add_node(label, type='label', feature=label_feat / norm)

        label_feat = np.zeros(3 * numeric_features_dim)
        # norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        G.add_node(label, type='label', feature=label_feat)

    for dataset_name, labels_info in dataset_ratios.items():
        G.add_node(dataset_name, type='dataset')
        total_feature = np.zeros(len(next(iter(label_features.values()))))
        total_weight = 0

        for label, ratio in labels_info.items():
            if label in label_features:
                weight = ratio
                feature = np.array(label_features[label])
                weighted_feature = weight * feature
                total_feature += weighted_feature
                total_weight += weight
                G.add_edge(dataset_name, label, weight=weight)

        if total_weight > 0:
            # Add norm
            feature = total_feature / total_weight
            num_feat = num_feat_dict[dataset_name]
            cat_feat = num_feat
            norm = np.linalg.norm(cat_feat, axis=-1, keepdims=True)
            G.nodes[dataset_name]['feature'] = cat_feat / norm
        else:
            G.nodes[dataset_name]['feature'] = np.zeros(len(total_feature) + 3 * numeric_features_dim)

    return G

def create_mean_graph(dataset_features, label_features, dataset_ratios, numeric_features):
    G = nx.DiGraph()

    numeric_features_dim = 256
    num_feats = np.array(
        [[data["num_images"], data["image_size"], data["num_classes"]] for data in numeric_features.values()])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(num_feats)
    expanded_features = np.repeat(normalized_features, numeric_features_dim, axis=1)
    num_feat_dict = {name: expanded_features[i] for i, name in enumerate(numeric_features.keys())}

    for label, features in label_features.items():
        # label_feat = np.array(features)
        # norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        # G.add_node(label, type='label', feature=label_feat / norm)

        label_feat = np.concatenate((np.array(features), np.zeros(3 * numeric_features_dim)))
        norm = np.linalg.norm(label_feat, axis=-1, keepdims=True)
        G.add_node(label, type='label', feature=label_feat / norm)

    for dataset_name, labels_info in dataset_ratios.items():
        G.add_node(dataset_name, type='dataset')
        total_feature = np.zeros(len(next(iter(label_features.values()))))
        total_weight = 0

        for label, ratio in labels_info.items():
            if label in label_features:
                weight = ratio
                feature = np.array(label_features[label])
                weighted_feature = weight * feature
                total_feature += weighted_feature
                total_weight += weight
                G.add_edge(dataset_name, label, weight=weight)

        feature = dataset_features[dataset_name]
        num_feat = num_feat_dict[dataset_name]
        cat_feat = np.concatenate((feature, num_feat))
        norm = np.linalg.norm(cat_feat, axis=-1, keepdims=True)
        G.nodes[dataset_name]['feature'] = cat_feat / norm

    return G



def draw_graph(G):
    plt.figure(figsize=(12, 12))
    pos = nx.kamada_kawai_layout(G)

    label_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 'label']
    dataset_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 'dataset']

    nx.draw_networkx_nodes(G, pos, nodelist=label_nodes, node_color='skyblue', node_size=50, label='Labels', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=dataset_nodes, node_color='orange', node_size=100, label='Datasets',
                           alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    plt.title('Graph of Datasets and Label Features')
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.show()


def save_graph(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)


def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


if __name__ == '__main__':
    # get_dataset_mean_features()
    # with open(DATASET_FEATURE_PATH, 'r') as f:
    #     dataset_feature = json.load(f)

    # get_label_features()
    # get_label_ratios()
    # get_numeric_features()

    label_features, dataset_ratios, numeric_features = load_data(FEATURE_OUTPUT_PATH, RATIO_OUTPUT_PATH, NUMERIC_FEATURE_OUTPUT_PATH)
    # dataset_features = np.load(DATASET_FEATURE_OUTPUT_PATH, allow_pickle=True).item()
    # print(label_features)
    # print(dataset_ratios)
    # print(numeric_features)
    G = create_numeric_graph(label_features, dataset_ratios, numeric_features)
    # G = create_numeric_graph(dataset_features ,label_features, dataset_ratios, numeric_features)
    save_graph(G, DATASET_GRAPH_OUTPUT_PATH)

    G = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    print(G)
    # draw_graph(G)

    # def update_dataset_vectors(G):
    #     numeric_features_dim = 256  # Assuming this dimension from your context
    #     for dataset_name, attr in G.nodes(data=True):
    #         if attr['type'] == 'dataset':
    #             print(1)
    #             total_feature = np.zeros(len(next(iter(G.nodes(data=True)))[1]['feature']) - 3 * numeric_features_dim)
    #             total_weight = 0
    #
    #             for neighbor, edge_attr in G[dataset_name].items():
    #                 if G.nodes[neighbor]['type'] == 'label':
    #                     weight = edge_attr['weight']
    #                     label_feature = G.nodes[neighbor]['feature'][:len(total_feature)]
    #                     weighted_feature = weight * label_feature
    #                     total_feature += weighted_feature
    #                     total_weight += weight
    #
    #             if total_weight > 0:
    #                 # Normalize and concatenate numeric features
    #                 feature = total_feature / total_weight
    #                 num_feat = G.nodes[dataset_name]['feature'][len(total_feature):]
    #                 cat_feat = np.concatenate((feature, num_feat))
    #                 norm = np.linalg.norm(cat_feat, axis=-1, keepdims=True)
    #                 G.nodes[dataset_name]['feature'] = cat_feat / norm
    #             else:
    #                 G.nodes[dataset_name]['feature'] = np.zeros(len(total_feature) + 3 * numeric_features_dim)
    #     return G
    #
    # s_t = time.time()
    # G = update_dataset_vectors(G)
    # e_t = time.time()
    # print(f'total time: {e_t - s_t}')
    #
