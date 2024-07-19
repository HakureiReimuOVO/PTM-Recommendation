import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv
from torch.nn import Linear, MSELoss, CrossEntropyLoss
from torch_geometric.data import Data
from tqdm import tqdm

from config import *
from dataset_graph import load_graph, DATASET_GRAPH_OUTPUT_PATH
from model_graph import MODEL_GRAPH_OUTPUT_PATH
from acc_loader import *
from train_test_indice import *
from evaluate_metrics import *
from slice_dataset import get_all_datasets_and_idx

model_save_path = 'saved_models'


def generate_combs(n):
    indices = list(range(n))
    combinations = list(itertools.combinations(indices, 2))
    return combinations


class DatasetGraphGCN(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(DatasetGraphGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, output_size)
        self.res_connection1 = Linear(num_features, 1024)
        self.res_connection2 = Linear(1024, output_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First convolution with residual connection
        x_res = self.res_connection1(x)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight) + x_res)
        x = F.dropout(x, training=self.training)

        # Second convolution with residual connection
        x_res = self.res_connection2(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight) + x_res
        return x


class ModelGraphGCN(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(ModelGraphGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, output_size)
        self.res_connection1 = Linear(num_features, 1024)
        self.res_connection2 = Linear(1024, output_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First convolution with residual connection
        x_res = self.res_connection1(x)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight) + x_res)
        x = F.dropout(x, training=self.training)

        # Second convolution with residual connection
        x_res = self.res_connection2(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight) + x_res
        return x


# class DatasetGraphGCN(torch.nn.Module):
#     def __init__(self, num_features, output_size):
#         super(DatasetGraphGCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 1024)
#         self.conv2 = GCNConv(1024, output_size)
#         # self.conv2 = GCNConv(1024, 2048)
#         # self.conv3 = GCNConv(2048, 4096)
#         # self.conv4 = GCNConv(4096, output_size)
#
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         return x
#
#         # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         # x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         # x = F.dropout(x, training=self.training)
#         # x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
#         # x = F.dropout(x, training=self.training)
#         # x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
#         # x = F.dropout(x, training=self.training)
#         # x = self.conv4(x, edge_index, edge_weight=edge_weight)
#         # return x
#
#
# class ModelGraphGCN(torch.nn.Module):
#     def __init__(self, num_features, output_size):
#         super(ModelGraphGCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 1024)
#         self.conv2 = GCNConv(1024, output_size)
#
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         return x


class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = Linear(input_dim, 1024)
        self.fc2 = Linear(1024, 2048)
        self.fc3 = Linear(2048, 1024)
        self.fc4 = Linear(1024, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        x = self.fc4(x)
        return x


# class RegressionModel(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RegressionModel, self).__init__()
#         self.fc1 = Linear(input_dim, 512)
#         self.fc2 = Linear(512, output_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x


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


def norm_acc(data):
    acc_values = []
    for dataset in data.values():
        for acc in dataset.values():
            acc_values.append(acc)

    acc_values = np.array(acc_values).reshape(-1, 1)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    standardized_acc_values = scaler.fit_transform(acc_values)

    index = 0
    for dataset in data.values():
        for model in dataset:
            dataset[model] = standardized_acc_values[index, 0]
            index += 1

    return scaler


if __name__ == '__main__':
    combs = generate_combs(len(model_configs))

    G_dataset = load_graph(DATASET_GRAPH_OUTPUT_PATH)
    G_model = load_graph(MODEL_GRAPH_OUTPUT_PATH)

    dataset_node_to_index = {node: idx for idx, node in enumerate(G_dataset.nodes())}
    model_node_to_index = {node: idx for idx, node in enumerate(G_model.nodes())}

    dataset_data = graph_to_torch_geometric(G_dataset)
    model_data = graph_to_torch_geometric(G_model)

    if fin_test:
        dataset_GCN = DatasetGraphGCN(dataset_data.num_node_features, 512)
        model_GCN = ModelGraphGCN(model_data.num_node_features, 512)
        regression_model = RegressionModel(1024, 1)

        epoch = 50

        dataset_GCN.load_state_dict(torch.load(os.path.join(model_save_path, f"dataset_GCN_epoch_{epoch}.pth")))
        model_GCN.load_state_dict(torch.load(os.path.join(model_save_path, f"model_GCN_epoch_{epoch}.pth")))
        regression_model.load_state_dict(
            torch.load(os.path.join(model_save_path, f"regression_model_epoch_{epoch}.pth")))

        input_tuples = []
        dataset_names = []

        for dataset_config in dataset_configs:
            items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
            for _, _, dataset_name in items:
                dataset_names.append(dataset_name)
                for model_name in model_configs:
                    input_tuples.append((dataset_name, model_name))

        acc = extract_accuracies('result/best_accuracies.csv')
        # Normalize
        scaler = norm_acc(acc)

        fin_tuples = []
        for input_tuple in input_tuples:
            if dataset_names.index(input_tuple[0]) in fin_indices:
                fin_tuples.append(input_tuple)

        # Fin test
        dataset_GCN.eval()
        model_GCN.eval()
        regression_model.eval()
        test_loss = 0

        with torch.no_grad():
            dataset_features = dataset_GCN(dataset_data)
            model_features = model_GCN(model_data)

            res_dict = {}

            for dataset_name, model_name in fin_tuples:
                f_dataset = dataset_features[dataset_node_to_index[dataset_name]]
                f_model = model_features[model_node_to_index[model_name]]

                combined_features = torch.cat([f_dataset, f_model], dim=0)

                target = torch.tensor([acc[dataset_name][model_name]], dtype=torch.float)
                output = regression_model(combined_features)

                real_target = scaler.inverse_transform(np.array(target).reshape(1, 1))
                real_output = scaler.inverse_transform(np.array(output).reshape(1, 1))

                if dataset_name not in res_dict:
                    res_dict[dataset_name] = {}

                res_dict[dataset_name][model_name] = real_output[0][0]

        print(res_dict)

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
        p_list = []

        binary_cnt = 0
        binary_acc = 0

        for k, v in res_dict.items():
            v_real = acc[k]
            v_predict = list(v.values())
            p_idx = v_predict.index(max(v.values()))

            v_real_scaled = scaler.inverse_transform(np.array(list(v_real.values())).reshape(-1, 1)).reshape(-1)
            p = v_real_scaled[p_idx]

            for comb in combs:
                p_x = v_predict[comb[0]]
                p_y = v_predict[comb[1]]
                r_x = v_real_scaled[comb[0]]
                r_y = v_real_scaled[comb[1]]
                binary_cnt += 1
                if r_x == r_y:
                    binary_acc += 1
                elif p_x > p_y and r_x > r_y:
                    binary_acc += 1
                elif p_x < p_y and r_x < r_y:
                    binary_acc += 1

            rmv_cnt += rmv(v_predict, v_real_scaled, 1)
            rmv2_cnt += rmv(v_predict, v_real_scaled, 2)
            rmv3_cnt += rmv(v_predict, v_real_scaled, 3)
            rmv4_cnt += rmv(v_predict, v_real_scaled, 4)
            rmv5_cnt += rmv(v_predict, v_real_scaled, 5)

            precision_cnt += precision_at_k(v_predict, v_real_scaled, 3)
            recall_cnt += recall_at_k(v_predict, v_real_scaled, 3)
            mrr_cnt += mrr_at_k(v_predict, v_real_scaled, 1)
            mrr2_cnt += mrr_at_k(v_predict, v_real_scaled, 2)
            mrr3_cnt += mrr_at_k(v_predict, v_real_scaled, 3)
            mrr4_cnt += mrr_at_k(v_predict, v_real_scaled, 4)
            mrr5_cnt += mrr_at_k(v_predict, v_real_scaled, 5)
            map_cnt += map_at_k(v_predict, v_real_scaled, 1)
            map2_cnt += map_at_k(v_predict, v_real_scaled, 2)
            map3_cnt += map_at_k(v_predict, v_real_scaled, 3)
            map4_cnt += map_at_k(v_predict, v_real_scaled, 4)
            map5_cnt += map_at_k(v_predict, v_real_scaled, 5)
            ndcg_cnt += ndcg_at_k(v_predict, v_real_scaled, 3)
            cnt += 1
            p_list.append((p_idx, p))

        print(p_list)
        # print(f'Average RMV: {sum(rmv_list) / len(rmv_list)}')
        # print(f'MAP: {sum(map_list) / len(map_list)}')
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
        print('=====================')
        pass
    else:
        dataset_GCN = DatasetGraphGCN(dataset_data.num_node_features, 512)
        model_GCN = ModelGraphGCN(model_data.num_node_features, 512)
        regression_model = RegressionModel(1024, 1)

        optimizer = Adam([
            {'params': dataset_GCN.parameters(), 'lr': 0.0001},
            {'params': model_GCN.parameters(), 'lr': 0.0001},
            {'params': regression_model.parameters(), 'lr': 0.0001}
        ])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        criterion = MSELoss()

        input_tuples = []
        dataset_names = []

        for dataset_config in dataset_configs:
            items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
            for _, _, dataset_name in items:
                dataset_names.append(dataset_name)
                for model_name in model_configs:
                    input_tuples.append((dataset_name, model_name))

        acc = extract_accuracies('result/best_accuracies.csv')
        # Normalize
        scaler = norm_acc(acc)

        # Test: Filter
        # filtered_tuples = []
        # for tuple in input_tuples:
        #     d_name = tuple[0]
        #     m_name = tuple[1]
        #     if tuple[0] in acc:
        #         if tuple[1] in acc[tuple[0]]:
        #             filtered_tuples.append(tuple)
        # input_tuples = filtered_tuples

        # train_tuples, test_tuples = train_test_split(input_tuples, test_size=0.2, random_state=42)

        train_tuples = []
        test_tuples = []
        for input_tuple in input_tuples:
            if dataset_names.index(input_tuple[0]) in train_indices:
                train_tuples.append(input_tuple)
            elif dataset_names.index(input_tuple[0]) in test_indices:
                test_tuples.append(input_tuple)

        # gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        # train_idx, test_idx = next(gss.split(input_tuples, groups=dataset_names))
        # train_tuples = [input_tuples[i] for i in train_idx]
        # test_tuples = [input_tuples[i] for i in test_idx]

        num_epochs = 200

        for epoch in range(num_epochs):
            # Train
            dataset_GCN.train()
            model_GCN.train()
            regression_model.train()
            train_loss = 0
            batch_loss = 0

            optimizer.zero_grad()

            dataset_features = dataset_GCN(dataset_data)
            model_features = model_GCN(model_data)

            for dataset_name, model_name in tqdm(train_tuples):
                f_dataset = dataset_features[dataset_node_to_index[dataset_name]]
                f_model = model_features[model_node_to_index[model_name]]

                combined_features = torch.cat([f_dataset, f_model], dim=0)

                target = torch.tensor([acc[dataset_name][model_name]], dtype=torch.float)
                output = regression_model(combined_features)

                loss = criterion(output, target)

                train_loss += loss.item()
                batch_loss += loss

            loss = train_loss / len(train_tuples)

            batch_loss.backward()
            optimizer.step()
            scheduler.step(loss)

            print(f'Epoch {epoch + 1}/{num_epochs}, Training loss: {loss}')

            if (epoch + 1) % 10 == 0:
                torch.save(dataset_GCN.state_dict(),
                           os.path.join(model_save_path, f"dataset_GCN_epoch_{epoch + 1}.pth"))
                torch.save(model_GCN.state_dict(), os.path.join(model_save_path, f"model_GCN_epoch_{epoch + 1}.pth"))
                torch.save(regression_model.state_dict(),
                           os.path.join(model_save_path, f"regression_model_epoch_{epoch + 1}.pth"))
                print(f'Model saved at epoch {epoch + 1}.')

            # Test
            dataset_GCN.eval()
            model_GCN.eval()
            regression_model.eval()
            test_loss = 0

            with torch.no_grad():
                dataset_features = dataset_GCN(dataset_data)
                model_features = model_GCN(model_data)

                res_dict = {}

                for dataset_name, model_name in test_tuples:
                    f_dataset = dataset_features[dataset_node_to_index[dataset_name]]
                    f_model = model_features[model_node_to_index[model_name]]

                    combined_features = torch.cat([f_dataset, f_model], dim=0)

                    target = torch.tensor([acc[dataset_name][model_name]], dtype=torch.float)
                    output = regression_model(combined_features)

                    real_target = scaler.inverse_transform(np.array(target).reshape(1, 1))
                    real_output = scaler.inverse_transform(np.array(output).reshape(1, 1))

                    if dataset_name not in res_dict:
                        res_dict[dataset_name] = {}

                    res_dict[dataset_name][model_name] = real_output[0][0]

                    loss = criterion(output, target)
                    test_loss += loss.item()

            print(res_dict)

            # Calculate MAP

            # Calculate RMV
            rmv_cnt = 0
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
            p_list = []

            binary_cnt = 0
            binary_acc = 0

            for k, v in res_dict.items():
                v_real = acc[k]
                v_predict = list(v.values())
                p_idx = v_predict.index(max(v.values()))

                v_real_scaled = scaler.inverse_transform(np.array(list(v_real.values())).reshape(-1, 1)).reshape(-1)
                p = v_real_scaled[p_idx]

                for comb in combs:
                    p_x = v_predict[comb[0]]
                    p_y = v_predict[comb[1]]
                    r_x = v_real_scaled[comb[0]]
                    r_y = v_real_scaled[comb[1]]
                    binary_cnt += 1
                    if r_x == r_y:
                        binary_acc += 1
                    elif p_x > p_y and r_x > r_y:
                        binary_acc += 1
                    elif p_x < p_y and r_x < r_y:
                        binary_acc += 1

                rmv_cnt += rmv(v_predict, v_real_scaled)
                precision_cnt += precision_at_k(v_predict, v_real_scaled, 3)
                recall_cnt += recall_at_k(v_predict, v_real_scaled, 3)
                mrr_cnt += mrr_at_k(v_predict, v_real_scaled, 1)
                mrr2_cnt += mrr_at_k(v_predict, v_real_scaled, 2)
                mrr3_cnt += mrr_at_k(v_predict, v_real_scaled, 3)
                mrr4_cnt += mrr_at_k(v_predict, v_real_scaled, 4)
                mrr5_cnt += mrr_at_k(v_predict, v_real_scaled, 5)
                map_cnt += map_at_k(v_predict, v_real_scaled, 1)
                map2_cnt += map_at_k(v_predict, v_real_scaled, 2)
                map3_cnt += map_at_k(v_predict, v_real_scaled, 3)
                map4_cnt += map_at_k(v_predict, v_real_scaled, 4)
                map5_cnt += map_at_k(v_predict, v_real_scaled, 5)

                ndcg_cnt += ndcg_at_k(v_predict, v_real_scaled, 3)
                cnt += 1
                p_list.append((p_idx, p))

            print(p_list)
            # print(f'Average RMV: {sum(rmv_list) / len(rmv_list)}')
            # print(f'MAP: {sum(map_list) / len(map_list)}')
            print(f'Average test loss: {test_loss / len(test_tuples)}')
            print(f'RMV: {rmv_cnt / cnt}')
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
            print('=====================')
