import json
import networkx as nx
import matplotlib.pyplot as plt

dataset_dump_path = '../result/dataset_features.json'
model_dump_path = '../result/model_features.json'

with open(dataset_dump_path, 'r') as f:
    dataset_features = json.load(f)

with open(model_dump_path, 'r') as f:
    model_features = json.load(f)

G = nx.Graph()

for dataset in dataset_features:
    G.add_node(dataset['dataset_name'], type='Dataset', **dataset)

for model in model_features:
    G.add_node(model['model_name'], type='Model', **model)

    if 'pretrained_dataset' in model and model['pretrained_dataset'] != 'None':
        G.add_edge(model['model_name'], model['pretrained_dataset'], type='pretrained_on')

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=500)

labels = {node: node for node in G.nodes() if 'type' in G.nodes[node]}

dataset_labels = {node: labels[node] for node in G.nodes() if
                  'type' in G.nodes[node] and G.nodes[node]['type'] == 'Dataset'}
model_labels = {node: labels[node] for node in G.nodes() if
                'type' in G.nodes[node] and G.nodes[node]['type'] == 'Model'}

nx.draw_networkx_labels(G, pos, labels=dataset_labels, font_size=12, font_color='green')
nx.draw_networkx_labels(G, pos, labels=model_labels, font_size=12, font_color='red')

# 绘制边
nx.draw_networkx_edges(G, pos)

# 显示图谱
plt.axis('off')
plt.show()
