import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

from model_loader import get_model
from config import model_configs

# from ptflops import get_model_complexity_info

TINY_IMAGENET_PATH = 'datasets/tiny_imagenet'
MODEL_GRAPH_OUTPUT_PATH = 'result/model_graph.pkl'
MODEL_FEATURE_OUTPUT_PATH = 'result/model_features.npy'
NUMERIC_FEATURE_OUTPUT_PATH = 'result/model_numeric_features.json'


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.jpeg') and not file.lower().startswith('.'):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def average_pool_to_fixed_length(feature, target_length=512):
    if feature.dim() != 1:
        raise ValueError("Feature must be a 1D tensor")

    feature = feature.unsqueeze(0).unsqueeze(0)
    pooled_feature = F.adaptive_avg_pool1d(feature, target_length)
    pooled_feature = pooled_feature.squeeze(0).squeeze(0)

    return pooled_feature


def forward_pass(data_loader, model, fc_layer):
    features = []
    outputs = []

    def hook_fn_forward(_, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if torch.cuda.is_available():
                data = data.cuda()
            _ = model(data)
    forward_hook.remove()

    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])

    return features, outputs


def extract_features(model_name, model, data_loader):
    if torch.cuda.is_available():
        model = model.cuda()

    # Different models has different linear projection names
    try:
        if model_name in ['microsoft/resnet-18', 'microsoft/resnet-50']:
            fc_layer = model.classifier[-1]
        else:
            fc_layer = model.classifier
    except Exception:
        raise NotImplementedError

    features, outputs = forward_pass(data_loader, model, fc_layer)

    mean_feature = features.mean(dim=0)
    return mean_feature


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_graph(model_names, model_features, numeric_features, model_similarities):
    G = nx.Graph()

    numeric_features_dim = 4
    num_feats = np.array([[data["params"], data["flops"]] for data in numeric_features.values()])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(num_feats)
    expanded_features = np.repeat(normalized_features, numeric_features_dim, axis=1)
    num_feat_dict = {name: expanded_features[i] for i, name in enumerate(numeric_features.keys())}

    for model_name in model_names:
        G.add_node(model_name)
        G.nodes[model_name]['features'] = np.concatenate((model_features[model_name], num_feat_dict[model_name]))
        # G.nodes[model_name]['feature'] = model_features[model_names]

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            weight = model_similarities[i, j]
            G.add_edge(model_names[i], model_names[j], weight=weight)

    return G


def save_graph(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)


def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def show_graph(G):
    pos = nx.spring_layout(G, k=0.5)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Model Similarity Graph")
    plt.show()


def get_model_features():
    global model_configs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNetDataset(root_dir=TINY_IMAGENET_PATH, transform=transform)

    print(len(dataset))

    # Test
    # dataset = torch.utils.data.Subset(dataset, range(32))

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model_features = {}

    # Test
    # model_configs = model_configs[:2]

    for model_config in model_configs:
        model = get_model(model_config)
        feature = extract_features(model_config, model, data_loader)
        pooled_feature = average_pool_to_fixed_length(feature, target_length=512)
        norm = np.linalg.norm(pooled_feature, axis=-1, keepdims=True)
        norm_feature = pooled_feature / norm

        model_features[model_config] = norm_feature.cpu().numpy()

        # params, trainable_params = get_model_params(model)
        # flops, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        # print(f'Total params: {params}, Flops: {flops}')

    np.save(MODEL_FEATURE_OUTPUT_PATH, model_features)


def load_data(feature_path, numeric_feature_path):
    features = np.load(feature_path, allow_pickle=True).item()

    with open(numeric_feature_path, 'r') as f:
        numeric_features = json.load(f)

    return features, numeric_features


# def get_model_numeric_features():
#     global model_configs
#     input_size = (3, 224, 224)
#     feat_dict = {}
#     for model_config in model_configs:
#         model = get_model(model_config)
#         params, trainable_params = get_model_params(model)
#         flops, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
#         feat_dict[model_config] = {
#             'params': params,
#             'flops': flops,
#         }
#
#         print(f'Total params: {params}, Flops: {flops}')
#
#     with open(NUMERIC_FEATURE_OUTPUT_PATH, 'w') as f:
#         json.dump(feat_dict, f)


if __name__ == '__main__':
    # get_model_features()
    # get_model_numeric_features()

    # model_features, numeric_features = load_data(MODEL_FEATURE_OUTPUT_PATH, NUMERIC_FEATURE_OUTPUT_PATH)
    #
    # model_features_matrix = np.array(list(model_features.values()))
    #
    # similarity_matrix = cosine_similarity(model_features_matrix)
    #
    # G = create_graph(model_names=model_configs, model_features=model_features, numeric_features=numeric_features,
    #                  model_similarities=similarity_matrix)
    #
    # save_graph(G, MODEL_GRAPH_OUTPUT_PATH)

    G = load_graph(MODEL_GRAPH_OUTPUT_PATH)
    print(G)
