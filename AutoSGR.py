import numpy as np

from dataset_graph import DATASET_GRAPH_OUTPUT_PATH, load_graph
from config import dataset_configs
from slice_dataset import get_all_datasets_and_idx

META_FEATURE_PATH = 'meta_features/AutoSGR'


def get_dataset_feature_vector(graph, dataset_name):
    if dataset_name in graph.nodes:
        node = graph.nodes[dataset_name]
        if 'feature' in node:
            return node['feature']
        else:
            print(f"Node '{dataset_name}' does not have 'features' attribute.")
            return None
    else:
        print(f"Node '{dataset_name}' not found in the graph.")
        return None


if __name__ == '__main__':
    G = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    for dataset_config in dataset_configs:
        items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])

        for _, _, dataset_name in items:
            meta_feature = get_dataset_feature_vector(G, dataset_name)
            np.save(f'{META_FEATURE_PATH}/{dataset_name}', meta_feature)
