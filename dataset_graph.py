import json
import pickle
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk
from config import *
from torchvision import models, transforms

from feature_extraction import get_dataset_features
from slice_dataset import get_all_datasets_and_idx

FEATURE_OUTPUT_PATH = 'result/label_features.npy'
RATIO_OUTPUT_PATH = 'result/label_ratio.json'
DATASET_GRAPH_OUTPUT_PATH = 'result/dataset_graph.pkl'


def get_dataset_feature(dataset_name, features):
    return next((feature for feature in features if feature["dataset_name"] == dataset_name), None)


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


def load_data(feature_path, ratio_path):
    label_features = np.load(feature_path, allow_pickle=True).item()

    with open(ratio_path, 'r') as f:
        dataset_ratios = json.load(f)

    # with open(feature_path, 'r') as f:
    #     label_features = json.load(f)

    return label_features, dataset_ratios


def create_graph(label_features, dataset_ratios):
    G = nx.DiGraph()

    for label, features in label_features.items():
        G.add_node(label, type='label', feature=np.array(features))

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
            norm = np.linalg.norm(feature, axis=-1, keepdims=True)
            G.nodes[dataset_name]['feature'] = feature / norm
        else:
            G.nodes[dataset_name]['feature'] = np.zeros(len(total_feature))

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
    # with open(DATASET_FEATURE_PATH, 'r') as f:
    #     dataset_feature = json.load(f)

    # get_label_features()
    # get_label_ratios()

    label_features, dataset_ratios = load_data(FEATURE_OUTPUT_PATH, RATIO_OUTPUT_PATH)
    print(label_features)
    print(dataset_ratios)
    G = create_graph(label_features, dataset_ratios)
    save_graph(G, DATASET_GRAPH_OUTPUT_PATH)

    # G = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    # print(G)
    # draw_graph(G)
