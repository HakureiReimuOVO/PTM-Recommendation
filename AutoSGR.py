from dataset_graph import GRAPH_OUTPUT_PATH, load_graph
from config import dataset_configs
from slice_dataset import get_all_datasets_and_idx


def get_dataset_feature_vector(graph, dataset_name):
    if dataset_name in graph.nodes:
        node = graph.nodes[dataset_name]
        if 'features' in node:
            return node['features']
        else:
            print(f"Node '{dataset_name}' does not have 'features' attribute.")
            return None
    else:
        print(f"Node '{dataset_name}' not found in the graph.")
        return None


if __name__ == '__main__':
    G = load_graph(GRAPH_OUTPUT_PATH)
    for dataset_config in dataset_configs:
        items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])

        for _, _, dataset_name in items:
            f = get_dataset_feature_vector(G, dataset_name)
            print(f'{dataset_name}: {f}')
