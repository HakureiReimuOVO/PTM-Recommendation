import csv
import os

import numpy as np
import pandas as pd

from config import dataset_configs
from feature_loader import get_mrm_features
from slice_dataset import get_all_datasets_and_idx


def format_dataset_name(dataset_name):
    parts = dataset_name.split('_')
    formatted_name = f"{parts[-2]}-{parts[-1]}"
    return formatted_name


def extract_dataset_accuracies(dataset_name, csv_folder):
    formatted_name = format_dataset_name(dataset_name)
    extracted_data = {}
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            df = pd.read_csv(file_path)
            extracted_rows = df[df['Task'] == formatted_name]
            if not extracted_rows.empty:
                model_name = extracted_rows['Architecture'].values[0]
                accuracy = extracted_rows['Accuracy_avg'].values[0]
                extracted_data[model_name] = accuracy
    return extracted_data


def extract_accuracies(csv_path):
    data_dict = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row['Dataset']
            model = row['Model']
            accuracy = float(row['Best Accuracy'])
            if dataset not in data_dict:
                data_dict[dataset] = {}
            data_dict[dataset][model] = accuracy
    return data_dict


def accuracies_to_regression_vector(accuracies):
    return np.array(list(accuracies.values()))


def accuracies_to_classification_vector(accuracies):
    acc_list = list(accuracies.values())
    max_acc = max(acc_list)
    idx = acc_list.index(max_acc)
    one_hot = np.zeros(len(list(accuracies.values())))
    one_hot[idx] = 1
    return one_hot


def accuracies_to_idx(accuracies):
    acc_list = list(accuracies.values())
    max_acc = max(acc_list)
    idx = acc_list.index(max_acc)
    return idx


if __name__ == '__main__':
    tmp = extract_accuracies('result/best_accuracies.csv')
    # for dataset_config in dataset_configs:
    #     items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
    #     for _, _, dataset_name in items:
    #         accuracies = extract_dataset_accuracies(dataset_name, 'cifar10_results')
    #         print(accuracies_to_classification_vector(accuracies, num_classes=6))
