import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetEncoder(nn.Module):
    def __init__(self, dataset_name_vocab_size, unique_labels_vocab_size, embedding_dim=10):
        super(DatasetEncoder, self).__init__()
        self.dataset_name_embedding = nn.Embedding(dataset_name_vocab_size, embedding_dim)
        self.labels_embedding = nn.Embedding(unique_labels_vocab_size, embedding_dim)

        # 假设所有的数值特征已被展平并且标准化
        num_numeric_features = 4  # 对于image_size, num_channels, num_samples, num_classes
        self.numeric_features_fc = nn.Linear(num_numeric_features, 20)

        # 合并嵌入层输出和数值特征的全连接层
        self.fc = nn.Linear(20 + 2 * embedding_dim, 64)  # 假设我们有两个嵌入向量和数值特征处理后的输出

    def forward(self, dataset_name, unique_labels, numeric_features):
        dataset_name_embedded = self.dataset_name_embedding(dataset_name)
        labels_embedded = self.labels_embedding(unique_labels).mean(dim=1)  # 假设unique_labels已经按索引编码

        numeric_features = self.numeric_features_fc(numeric_features)

        # 合并特征
        features = torch.cat([dataset_name_embedded, labels_embedded, numeric_features], dim=1)
        output = F.relu(self.fc(features))
        return output


class ModelEncoder(nn.Module):
    def __init__(self, model_name_vocab_size, pretrained_dataset_vocab_size, model_type_vocab_size,
                 model_owner_vocab_size, model_architecture_vocab_size, model_task_vocab_size, embedding_dim=10):
        super(ModelEncoder, self).__init__()
        self.model_name_embedding = nn.Embedding(model_name_vocab_size, embedding_dim)
        self.pretrained_dataset_embedding = nn.Embedding(pretrained_dataset_vocab_size, embedding_dim)
        self.model_type_embedding = nn.Embedding(model_type_vocab_size, embedding_dim)
        self.model_owner_embedding = nn.Embedding(model_owner_vocab_size, embedding_dim)
        self.model_architecture_embedding = nn.Embedding(model_architecture_vocab_size, embedding_dim)
        self.model_task_embedding = nn.Embedding(model_task_vocab_size, embedding_dim)

        # 数值特征的处理：假设所有的数值特征已被展平并且标准化
        num_numeric_features = 6  # 对于num_parameters, num_layers, input_size (两个维度), num_classes, downloads
        self.numeric_features_fc = nn.Linear(num_numeric_features, 20)

        # 合并嵌入层输出和数值特征的全连接层
        total_embedding_size = embedding_dim * 6  # 有六个分类特征的嵌入
        self.fc = nn.Linear(20 + total_embedding_size, 64)  # 合并后的特征通过全连接层

    def forward(self, model_name, pretrained_dataset, model_type, model_owner, model_architecture, model_task,
                numeric_features):
        model_name_embedded = self.model_name_embedding(model_name)
        pretrained_dataset_embedded = self.pretrained_dataset_embedding(pretrained_dataset)
        model_type_embedded = self.model_type_embedding(model_type)
        model_owner_embedded = self.model_owner_embedding(model_owner)
        model_architecture_embedded = self.model_architecture_embedding(model_architecture)
        model_task_embedded = self.model_task_embedding(model_task)

        numeric_features = self.numeric_features_fc(numeric_features)

        features = torch.cat(
            [model_name_embedded, pretrained_dataset_embedded, model_type_embedded, model_owner_embedded,
             model_architecture_embedded, model_task_embedded, numeric_features], dim=1)
        output = F.relu(self.fc(features))
        return output


class ScoreDecoder(nn.Module):
    def __init__(self, input_dim_dataset, input_dim_model, hidden_dim=64):
        super(ScoreDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim_dataset + input_dim_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)  # 输出层，预测一个回归分数

    def forward(self, encoded_dataset, encoded_model):
        combined_features = torch.cat((encoded_dataset, encoded_model), dim=1)

        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        score = self.output(x)  # 不使用激活函数，直接输出分数

        return score


class PTM(nn.Module):
    def __init__(self, dataset_encoder, model_encoder, decoder):
        super(PTM, self).__init__()
        self.dataset_encoder = dataset_encoder
        self.model_encoder = model_encoder
        self.decoder = decoder

    def forward(self, dataset_features, model_features):
        encoded_dataset = self.dataset_encoder(
            dataset_features['dataset_name'],
            dataset_features['unique_labels'],
            dataset_features['numeric_features']
        )

        encoded_model = self.model_encoder(
            model_features['model_name'],
            model_features['pretrained_dataset'],
            model_features['model_type'],
            model_features['model_owner'],
            model_features['model_architecture'],
            model_features['model_task'],
            model_features['numeric_features']
        )

        predicted_score = self.decoder(encoded_dataset, encoded_model)
        return predicted_score


dataset_encoder = DatasetEncoder(...)
model_encoder = ModelEncoder(...)
decoder = ScoreDecoder(...)

model = PTM(dataset_encoder, model_encoder, decoder)

dataset_features = {
    'dataset_name': ...,
    'unique_labels': ...,
    'numeric_features': ...
}

model_features = {
    'model_name': ...,
    'pretrained_dataset': ...,
    'model_type': ...,
    'model_owner': ...,
    'model_architecture': ...,
    'model_task': ...,
    'numeric_features': ...
}

predicted_score = model(dataset_features, model_features)
print(predicted_score)
