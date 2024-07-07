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
from model_graph import MODEL_GRAPH_OUTPUT_PATH
from acc_loader import *


class DatasetGraphGCN(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(ModelGraphGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 2048)
        self.conv3 = GCNConv(2048, 4096)
        self.conv4 = GCNConv(4096, output_size)

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


class ModelGraphGCN(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(ModelGraphGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, output_size)

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


class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = Linear(input_dim, 512)
        self.fc2 = Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def graph_to_torch_geometric(G):
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
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

    G_dataset = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    G_model = load_graph(MODEL_GRAPH_OUTPUT_PATH)

    score_dict = {}
    for node in G_dataset.nodes():
        if not G_dataset.nodes[node]['type'] == 'label':
            acc = extract_accuracies(node, 'cifar10_results')
            # vec = accuracies_to_regression_vector(acc)
            vec = accuracies_to_classification_vector(acc)
            score_dict[node] = vec

    print(score_dict)

    node_to_index = {node: idx for idx, node in enumerate(G_dataset.nodes())}
    dataset_data = graph_to_torch_geometric(G_dataset)
    model_data = graph_to_torch_geometric(G_model)

    dataset_GCN = DatasetGraphGCN(dataset_data.num_node_features, 512)
    model_GCN = ModelGraphGCN(model_data.num_node_features, 512)
    regression_model = RegressionModel(1024, 1)

    optimizer = Adam([
        {'params': dataset_GCN.parameters(), 'lr': 0.001},
        {'params': model_GCN.parameters(), 'lr': 0.001},
        {'params': regression_model.parameters(), 'lr': 0.001}
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    criterion = MSELoss()

    train_mask, test_mask = split_data(G_dataset)

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion, train_mask)
        accuracy, rmv = test(model, data, test_mask)
        print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%, RMV: {rmv * 100}%')
        scheduler.step(loss)

    # =======================================================================================
    #
    # node_to_index = {node: idx for idx, node in enumerate(G_dataset.nodes())}
    # unique_types = set(G_dataset.nodes[node]['type'] for node in G_dataset.nodes())
    # type_to_index = {type: idx for idx, type in enumerate(unique_types)}
    #
    # score_dict = {}
    # for node in G_dataset.nodes():
    #     if not G_dataset.nodes[node]['type'] == 'label':
    #         acc = extract_accuracies(node, 'cifar10_results')
    #         vec = accuracies_to_regression_vector(acc)
    #         # vec = accuracies_to_classification_vector(acc, num_classes=6)
    #         score_dict[node] = vec
    #
    # data = from_networkx_to_torch_geometric(G_dataset, node_to_index, type_to_index, score_dict, num_classes=6)
    #
    # model = ModelGraphGCN(num_features=data.num_node_features, output_size=6)
    # optimizer = Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    #
    # criterion = MSELoss()
    # # criterion = CrossEntropyLoss()
    #
    # # train_mask = torch.tensor([node_data['type'] == 'dataset' for node_id, node_data in G.nodes(data=True)],
    # #                           dtype=torch.bool)
    #
    # train_mask, test_mask = split_data(G_dataset)
    #
    # for epoch in range(200):
    #     loss = train(model, data, optimizer, criterion, train_mask)
    #     accuracy, rmv = test(model, data, test_mask)
    #     print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%, RMV: {rmv * 100}%')
    #     scheduler.step(loss)
