import itertools

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
from evaluate_metrics import *
from train_test_indice import *

model_save_path = 'saved_models'


def generate_combs(n):
    indices = list(range(n))
    combinations = list(itertools.combinations(indices, 2))
    return combinations


combs = generate_combs(len(model_configs))


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 2048)
        self.conv3 = GCNConv(2048, 4096)
        self.conv4 = GCNConv(4096, 2048)
        self.out = Linear(2048, num_classes)

        self.res1 = Linear(num_features, 1024)
        self.res2 = Linear(1024, 2048)
        self.res3 = Linear(2048, 4096)
        self.res4 = Linear(4096, 2048)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        res_x = self.res1(x)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight) + res_x)
        x = F.dropout(x, training=self.training)

        res_x = self.res2(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight) + res_x)
        x = F.dropout(x, training=self.training)

        res_x = self.res3(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight) + res_x)
        x = F.dropout(x, training=self.training)

        res_x = self.res4(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight) + res_x)

        x = self.out(x)
        return x


# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 1024)
#         self.conv2 = GCNConv(1024, 2048)
#         self.conv3 = GCNConv(2048, 4096)
#         self.conv4 = GCNConv(4096, 2048)
#         self.out = Linear(2048, num_classes)
#
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#
#         x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
#
#         x = self.out(x)
#         return x


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
        real = data.y[test_mask]

        rmv_cnt = 0
        rmv2_cnt = 0
        rmv3_cnt = 0
        rmv4_cnt = 0
        rmv5_cnt = 0
        precision_cnt = 0
        recall_cnt = 0
        mrr_cnt = 0
        mrr2_cnt = 0
        mrr3_cnt = 0
        mrr4_cnt = 0
        mrr5_cnt = 0
        map_cnt = 0
        map2_cnt = 0
        map3_cnt = 0
        map4_cnt = 0
        map5_cnt = 0
        ndcg_cnt = 0
        cnt = 0

        binary_acc = 0
        binary_cnt = 0

        for idx in range(out.shape[0]):
            p = out[idx].numpy()
            r = real[idx].numpy()

            for comb in combs:
                p_x = p[comb[0]]
                p_y = p[comb[1]]
                r_x = r[comb[0]]
                r_y = r[comb[1]]
                binary_cnt += 1
                if r_x == r_y:
                    binary_acc += 1
                elif p_x > p_y and r_x > r_y:
                    binary_acc += 1
                elif p_x < p_y and r_x < r_y:
                    binary_acc += 1

            cnt += 1
            rmv_cnt += rmv(p, r, 1)
            rmv2_cnt += rmv(p, r, 2)
            rmv3_cnt += rmv(p, r, 3)
            rmv4_cnt += rmv(p, r, 4)
            rmv5_cnt += rmv(p, r, 5)

            precision_cnt += precision_at_k(p, r, 3)
            recall_cnt += recall_at_k(p, r, 3)

            mrr_cnt += mrr_at_k(p, r, 1)
            mrr2_cnt += mrr_at_k(p, r, 2)
            mrr3_cnt += mrr_at_k(p, r, 3)
            mrr4_cnt += mrr_at_k(p, r, 4)
            mrr5_cnt += mrr_at_k(p, r, 5)

            map_cnt += map_at_k(p, r, 1)
            map2_cnt += map_at_k(p, r, 2)
            map3_cnt += map_at_k(p, r, 3)
            map4_cnt += map_at_k(p, r, 4)
            map5_cnt += map_at_k(p, r, 5)

            ndcg_cnt += ndcg_at_k(p, r, 3)

        print(f'RMV: {rmv_cnt / cnt}')
        print(f'RMV2: {rmv2_cnt / cnt}')
        print(f'RMV3: {rmv3_cnt / cnt}')
        print(f'RMV4: {rmv4_cnt / cnt}')
        print(f'RMV5: {rmv5_cnt / cnt}')

        print(f'Precision: {precision_cnt / cnt}')
        print(f'Recall: {recall_cnt / cnt}')
        print(f'MRR: {mrr_cnt / cnt}')
        print(f'MRR2: {mrr2_cnt / cnt}')
        print(f'MRR3: {mrr3_cnt / cnt}')
        print(f'MRR4: {mrr4_cnt / cnt}')
        print(f'MRR5: {mrr5_cnt / cnt}')

        print(f'MAP: {map_cnt / cnt}')
        print(f'MAP2: {map2_cnt / cnt}')
        print(f'MAP3: {map3_cnt / cnt}')
        print(f'MAP4: {map4_cnt / cnt}')
        print(f'MAP5: {map5_cnt / cnt}')

        print(f'NDCG: {ndcg_cnt / cnt}')
        print(f'binary acc: {binary_acc / binary_cnt}')
        print(f'=================================')

        # out = model(data)[test_mask]
        # predicted = out.argmax(dim=1)
        # correct = data.y[test_mask].argmax(dim=1)
        # correct_sum = (predicted == correct).sum().item()
        # total = test_mask.sum().item()
        # accuracy = correct_sum / total if total > 0 else 0
        #
        # # Caculate RMV
        # real_acc = data.y[test_mask]
        # max_acc = torch.max(real_acc, dim=1).values
        # predicted_tmp = predicted.unsqueeze(1)
        # predicted_acc = torch.gather(real_acc, 1, predicted_tmp).squeeze(1)
        # rmv = predicted_acc / max_acc
        # avg_rmv = torch.mean(rmv)


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
            if node in accuracies:
                acc = accuracies[node]
                vec = accuracies_to_regression_vector(acc)
                # vec = accuracies_to_classification_vector(acc)
                score_dict[node] = vec

    data = from_networkx_to_torch_geometric(G, node_to_index, type_to_index, score_dict, num_classes=8)

    label_cnt = 0
    for node in G.nodes():
        if G.nodes[node]['type'] == 'label':
            label_cnt += 1

    if fin_test:
        model = GCN(num_features=data.num_node_features, num_classes=8)
        epoch = 100
        model.load_state_dict(torch.load(os.path.join(model_save_path, f"classification_GCN_epoch_{epoch}.pth")))
        fin_mask = [False for idx in range(len(G.nodes))]
        for fin_indice in fin_indices:
            fin_mask[fin_indice + label_cnt] = True
        fin_mask = torch.tensor(fin_mask, dtype=torch.bool)
        test(model, data, fin_mask)
    else:
        model = GCN(num_features=data.num_node_features, num_classes=8)
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        criterion = MSELoss()
        # criterion = CrossEntropyLoss()

        # train_mask = torch.tensor([node_data['type'] == 'dataset' for node_id, node_data in G.nodes(data=True)],
        #                           dtype=torch.bool)

        # train_mask, test_mask = split_data(G)

        train_mask = [False for idx in range(len(G.nodes))]
        test_mask = [False for idx in range(len(G.nodes))]
        for train_indice in train_indices:
            train_mask[train_indice + label_cnt] = True
        for test_indice in test_indices:
            test_mask[test_indice + label_cnt] = True
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        for epoch in range(200):
            loss = train(model, data, optimizer, criterion, train_mask)
            test(model, data, test_mask)
            print(f'Epoch {epoch + 1}: Loss {loss:.4f}')
            # print(f'Epoch {epoch + 1}: Loss {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%, RMV: {rmv * 100}%')
            scheduler.step(loss)
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_save_path, f"classification_GCN_epoch_{epoch + 1}.pth"))
                print(f'Model saved at epoch {epoch + 1}.')
