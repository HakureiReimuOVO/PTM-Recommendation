import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch.nn import Linear, MSELoss
from torch_geometric.data import Data
from config import model_configs
from dataset_graph import load_graph, DATASET_GRAPH_OUTPUT_PATH


def load_model_scores(directory):
    dataset_scores = {}

    for filename in os.listdir(directory):
        if filename.endswith("_score_dict.json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                scores = json.load(file)
                dataset_name = filename.replace("_score_dict.json", "")
                dataset_scores[dataset_name] = scores

    return dataset_scores


def from_networkx_to_torch_geometric(G, node_to_index, type_to_index, score_dict):
    edge_index = []
    edge_weight = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_weight.append(data.get('weight', 1.0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    x = []
    for node_id, node_data in G.nodes(data=True):
        x.append(node_data['features'])
    x = torch.tensor(x, dtype=torch.float)

    y = []
    for node_id, node_data in G.nodes(data=True):
        if node_to_index[node_id] in score_dict:
            score = score_dict[node_to_index[node_id]]
            y.append(score_dict_to_array(score))
        else:
            y.append([0] * len(model_to_index))

    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    return data


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 2048)
        self.conv3 = GCNConv(2048, 4096)
        self.conv4 = GCNConv(4096, 2048)
        self.out = Linear(2048, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, edge_index))

        x = self.out(x)
        # return F.log_softmax(x, dim=1)
        return x


def score_dict_to_array(model_scores_dict):
    model_to_index = {
        model: idx for idx, model in enumerate(sorted(model_scores_dict.keys()))
    }
    scores_array = [0] * len(model_to_index)
    for model, score in model_scores_dict.items():
        index = model_to_index[model]
        scores_array[index] = score
    return scores_array


def load_and_transform_scores(directory, node_to_index, model_to_index):
    dataset_scores = {}
    for filename in os.listdir(directory):
        if filename.endswith("_score_dict.json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                scores = json.load(file)
                dataset_name = filename.replace("_score_dict.json", "")
                dataset_index = node_to_index[dataset_name]

                # scores_array = [scores.get(model, 0) for model in model_to_index]  # 默认分数为0
                # scores_tensor = torch.tensor(scores_array)
                # softmax_scores = torch.softmax(scores_tensor, dim=0)
                # transformed_scores = {model_to_index[model]: softmax_scores[i].item() for i, model in
                #                       enumerate(model_to_index)}

                # transformed_scores = {
                #     model_to_index[model]: score / max(scores.values())
                #     for model, score in scores.items()
                #     if model in model_to_index
                # }

                scores_array = [scores.get(model, 0) for model in model_to_index]  # 默认分数为0
                min_score = min(scores_array)
                max_score = max(scores_array)
                transformed_scores = {
                    model_to_index[model]: (score - min_score) / (max_score - min_score)
                    for model, score in scores.items()
                    if model in model_to_index and max_score > min_score
                }

                dataset_scores[dataset_index] = transformed_scores
    return dataset_scores


def test(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data)[test_mask]
        predicted = out.argmax(dim=1)
        correct = (predicted == data.y[test_mask].argmax(dim=1)).sum().item()
        total = test_mask.sum().item()
        accuracy = correct / total if total > 0 else 0
    return accuracy


def train(model, data, optimizer, criterion, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)[train_mask]
    loss = criterion(out, data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def split_data(G, test_ratio=0.2):
    test_mask = torch.zeros(len(G.nodes()), dtype=torch.bool)
    train_mask = torch.zeros(len(G.nodes()), dtype=torch.bool)

    dataset_indices = [idx for idx, (node_id, node_data) in enumerate(G.nodes(data=True)) if
                       node_data['type'] == 'dataset']

    np.random.shuffle(dataset_indices)
    num_test = int(len(dataset_indices) * test_ratio)

    test_indices = dataset_indices[:num_test]
    train_indices = dataset_indices[num_test:]

    test_mask[test_indices] = True
    train_mask[train_indices] = True

    return train_mask, test_mask


if __name__ == '__main__':
    model_to_index = {model: idx for idx, model in enumerate(model_configs)}

    G = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    score_dict = load_and_transform_scores('result', node_to_index, model_to_index)
    unique_types = set(G.nodes[node]['type'] for node in G.nodes())
    type_to_index = {type: idx for idx, type in enumerate(unique_types)}

    data = from_networkx_to_torch_geometric(G, node_to_index, type_to_index, score_dict)

    model = GCN(num_features=data.num_node_features, num_classes=len(model_configs))
    optimizer = Adam(model.parameters(), lr=0.00001)
    criterion = MSELoss()

    # train_mask = torch.tensor([node_data['type'] == 'dataset' for node_id, node_data in G.nodes(data=True)],
    #                           dtype=torch.bool)

    train_mask, test_mask = split_data(G)

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion, train_mask)
        accuracy = test(model, data, test_mask)
        print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # num_features = data.num_node_features
    # num_classes = len(model_configs)
    #
    # model = GCN(num_features, num_classes)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # criterion = torch.nn.NLLLoss()

    # def train():
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 只在训练节点上计算损失
    #     loss.backward()
    #     optimizer.step()
    #     return loss
    #
    # for epoch in range(200):
    #     loss = train()
    #     print(f'Epoch {epoch+1}: Loss: {loss.item()}')
    #
    # model.eval()
    # with torch.no_grad():
    #     out = model(data)
    #     index = 0
    #     predicted_model = torch.argmax(out[index]).item()
    #     recommended_model = model_configs[predicted_model]
    #     print(f'Recommended Model for Node {index}: {recommended_model}')

    # import torch
    # import torch.nn.functional as F
    # from torch_geometric.data import Data
    # from torch_geometric.nn import GCNConv
    # from dataset_graph import load_graph, GRAPH_OUTPUT_PATH
    #
    #
    # def from_networkx_to_torch_geometric(G):
    #     edge_index = torch.tensor(list(map(list, zip(*G.edges))), dtype=torch.long).t().contiguous()
    #     x = torch.tensor([G.nodes[n]['features'] for n in G.nodes], dtype=torch.float)
    #     y = torch.tensor([G.nodes[n].get('label', 0) for n in G.nodes], dtype=torch.long)
    #     train_mask = torch.tensor([G.nodes[n]['type'] == 'dataset' for n in G.nodes], dtype=torch.bool)
    #     data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
    #     return data
    #
    #
    # class GCN(torch.nn.Module):
    #     def __init__(self, num_features, num_classes):
    #         super(GCN, self).__init__()
    #         self.conv1 = GCNConv(num_features, 16)
    #         self.conv2 = GCNConv(16, num_classes)
    #
    #     def forward(self, data):
    #         x, edge_index = data.x, data.edge_index
    #         x = F.relu(self.conv1(x, edge_index))
    #         x = F.dropout(x, training=self.training)
    #         x = self.conv2(x, edge_index)
    #         return F.log_softmax(x, dim=1)
    #
    #
    # if __name__ == '__main__':
    #     G = load_graph(GRAPH_OUTPUT_PATH)
    #     data = from_networkx_to_torch_geometric(G)
    #
    #     num_classes = 2
    #     model = GCN(num_features=data.num_node_features, num_classes=num_classes)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #     criterion = torch.nn.NLLLoss()
    #
    #
    #     def train():
    #         model.train()
    #         optimizer.zero_grad()
    #         out = model(data)
    #         loss = criterion(out[data.train_mask], data.y[data.train_mask])
    #         loss.backward()
    #         optimizer.step()
    #         return loss
    #
    #
    #     for epoch in range(200):
    #         loss = train()
    #         print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
