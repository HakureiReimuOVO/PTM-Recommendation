import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv
from torch.nn import Linear, MSELoss, CrossEntropyLoss
from torch_geometric.data import Data
from config import model_configs
from dataset_graph import load_graph, DATASET_GRAPH_OUTPUT_PATH
from acc_loader import *


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 2048)
        self.conv3 = GCNConv(2048, 4096)
        self.conv4 = GCNConv(4096, 2048)
        self.out = Linear(2048, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))

        x = self.out(x)
        return x


def from_networkx_to_torch_geometric(G, node_to_index, type_to_index, score_dict, num_classes=6):
    edge_index = []
    edge_weight = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_weight.append(data.get('weight', 1.0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    x = []
    for node_id, node_data in G.nodes(data=True):
        x.append(node_data['feature'])
    x = torch.tensor(x, dtype=torch.float)

    y = []
    for node_id, node_data in G.nodes(data=True):
        if node_id in score_dict:
            score = score_dict[node_id]
            y.append(score)
        else:
            y.append([0] * num_classes)

    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    return data


def test(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data)[test_mask]
        predicted = out.argmax(dim=1)
        correct = data.y[test_mask].argmax(dim=1)
        correct_sum = (predicted == correct).sum().item()
        total = test_mask.sum().item()
        accuracy = correct_sum / total if total > 0 else 0

        # Caculate RMV
        real_acc = data.y[test_mask]
        max_acc = torch.max(real_acc, dim=1).values
        predicted_tmp = predicted.unsqueeze(1)
        predicted_acc = torch.gather(real_acc, 1, predicted_tmp).squeeze(1)
        rmv = predicted_acc / max_acc
        avg_rmv = torch.mean(rmv)

    return accuracy, avg_rmv


def train(model, data, optimizer, criterion, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)[train_mask]
    # loss = criterion(out, data.y[train_mask].argmax(dim=1))
    correct = data.y[train_mask]
    loss = criterion(out, correct)
    loss.backward()
    optimizer.step()
    return loss.item()


def split_data(G, test_ratio=0.2, seed=40):
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    # model_to_index = {model: idx for idx, model in enumerate(model_configs)}

    G = load_graph(DATASET_GRAPH_OUTPUT_PATH)

    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    unique_types = set(G.nodes[node]['type'] for node in G.nodes())
    type_to_index = {type: idx for idx, type in enumerate(unique_types)}

    score_dict = {}
    accuracies = extract_accuracies('result/best_accuracies.csv')
    for node in G.nodes():
        if not G.nodes[node]['type'] == 'label':
            # acc = extract_dataset_accuracies(node, 'cifar10_results')
            acc = accuracies[node]
            vec = accuracies_to_regression_vector(acc)
            # vec = accuracies_to_classification_vector(acc)
            score_dict[node] = vec

    data = from_networkx_to_torch_geometric(G, node_to_index, type_to_index, score_dict, num_classes=8)

    model = GCN(num_features=data.num_node_features, num_classes=8)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    criterion = MSELoss()
    # criterion = CrossEntropyLoss()

    # train_mask = torch.tensor([node_data['type'] == 'dataset' for node_id, node_data in G.nodes(data=True)],
    #                           dtype=torch.bool)

    train_mask, test_mask = split_data(G)

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion, train_mask)
        accuracy, rmv = test(model, data, test_mask)
        print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%, RMV: {rmv * 100}%')
        scheduler.step(loss)
