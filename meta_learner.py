import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier
from acc_loader import *
import metric
from config import *
from slice_dataset import get_all_datasets_and_idx
from feature_loader import *

num_tasks = 160
feature_length = 512

# Models count
num_classes = 10

wd_list = []
y_list = []

datasets = []
models = model_configs

accuracies = extract_accuracies('result/best_accuracies.csv')
for dataset_config in dataset_configs:
    items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
    for _, _, dataset_name in items:
        datasets.append(dataset_name)
        # AutoSGR
        meta_feature = get_sgr_features(dataset_name)
        # AutoMRM
        # meta_feature = get_mrm_features(dataset_name)

        acc = accuracies[dataset_name]
        vec = accuracies_to_idx(acc)

        wd_list.append(meta_feature)
        y_list.append(vec)

label_encoder = LabelEncoder()
y_list = label_encoder.fit_transform(y_list)
unique_y = label_encoder.classes_

wd_list = np.array(wd_list)
y_list = np.array(y_list)

# y_list = np.random.randint(0, num_classes, size=wd_list.shape[0])
# label_encoder = LabelEncoder()
# y_list = label_encoder.fit_transform(y_list)
# base_labels = np.arange(num_classes)
# remaining_labels = wd_list.shape[0] - num_classes
# random_labels = np.random.randint(0, num_classes, size=remaining_labels)
# y_list = np.concatenate([base_labels, random_labels])
# np.random.shuffle(y_list)

# (45, 2048)
print(wd_list.shape)

# (45)
print(y_list.shape)
print(y_list)


class Model(object):
    def __init__(self, name, SDS_acc, real_acc):
        self.name = name
        self.SDS_acc = SDS_acc
        self.real_acc = real_acc


#  Split the training and testing sets into 4:1 and repeat 100 times to obtain experimental results
acc_sum = 0.0
acc_list = []
ndcg5_t = []
mrr_t = []
map_t = []

# Select the meta learner to use Ensemble learning or not according to the task
model = RandomForestClassifier()
# model = BaggingClassifier(base_estimator=SVC(probability=True), max_samples=0.8, max_features=0.8, n_estimators=100,
#                           bootstrap_features=True, n_jobs=-1)
pipe_lr = make_pipeline(StandardScaler(), model)

res = []

for i in range(100):
    indices = np.arange(y_list.shape[0])
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, y_list, test_size=0.20, stratify=y_list)
    X_train_list = []
    X_test_list = []
    for j in indices_train:
        X_train_list.append(wd_list[j])
    X_train = np.array(X_train_list)
    for k in indices_test:
        X_test_list.append(wd_list[k])
    X_test = np.array(X_test_list)
    pipe_lr.fit(X_train, y_train)
    prediction_pro = pipe_lr.predict_proba(X_test)

    # res = df.iloc[indices_test]
    # res = res.reset_index(drop=True)

    ndcg5 = []
    mrr = []
    map = []
    rmv_list = []
    for x in range(len(prediction_pro)):
        model_list = []
        m = max(prediction_pro[x])
        a = list(prediction_pro[x]).index(m)
        idx = unique_y[a]
        model = models[idx]
        dataset = datasets[x]
        acc = accuracies[dataset][model]
        max_acc = max(accuracies[dataset].values())
        rmv_list.append(acc / max_acc)

        arr = np.array(prediction_pro)
        print(np.argmax(arr, axis=1))

        for y in range(len(prediction_pro[0])):
            model_name = y
            mSDS_acc = prediction_pro[x][y]
            #   mReal_acc = res[y][x]
            mReal_acc = 1

            model = Model(name=model_name, SDS_acc=mSDS_acc, real_acc=mReal_acc)
            model_list.append(model)
        # ndcg5.append(metric.NDCG(model_list, 5, 1)[0])
        # mrr.append(metric.MRR(model_list, 1)[0])
        # map.append(metric.MAP(model_list, 3, 1)[0])
    tmp = pipe_lr.score(X_test, y_test)
    acc_sum += tmp
    acc_list.append(tmp)
    print(f'epoch{i}, acc: {tmp}, rmv: {sum(rmv_list) / len(rmv_list)}')
    # ndcg5_t.append(sum(ndcg5) / len(ndcg5))
    # mrr_t.append(sum(mrr) / len(mrr))
    # map_t.append(sum(map) / len(map))

    res.append(sum(rmv_list) / len(rmv_list))

acc_avg = acc_sum / len(acc_list)
print("acc: ", acc_avg)
res_avg = sum(res) / len(res)
print("final rmv: ", res_avg)

# print("ndcg5: ", sum(ndcg5_t) / len(ndcg5_t))
# print("mrr: ", sum(mrr_t) / len(mrr_t))
# print("map: ", sum(map_t) / len(map_t))

# Save experimental results
# with open('acc_list_10t.pkl', 'wb') as e1:
#     pickle.dump(acc_list, e1)

# with open('ndcg5_list_10t.pkl', 'wb') as e2:
#     pickle.dump(ndcg5_t, e2)

# with open('mrr_list_10t.pkl', 'wb') as e2:
#     pickle.dump(mrr_t, e2)
#
# with open('map_list_10t.pkl', 'wb') as e2:
#     pickle.dump(map_t, e2)
