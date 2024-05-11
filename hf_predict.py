import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx

# 创建NetworkX图
G = nx.Graph()
G.add_node('dataset1', type='dataset', vector=np.random.rand(10))
G.add_node('label1', type='label', vector=np.random.rand(10))
G.add_node('label2', type='label', vector=np.random.rand(10))
G.add_edge('dataset1', 'label1')
G.add_edge('dataset1', 'label2')


class GCN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


def prepare_data(graph):
    for node in graph.nodes(data=True):
        graph.nodes[node[0]]['x'] = torch.tensor(node[1]['vector'], dtype=torch.float)

    data = from_networkx(graph)
    return data

feature_size = 10
hidden_size = 16
gnn_model = GCN(feature_size, hidden_size)

data = prepare_data(G)


gnn_model.eval()
with torch.no_grad():
    aggregated_features = gnn_model(data)
    print("Aggregated Features:", aggregated_features)
