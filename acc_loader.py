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


def extract_accuracies(dataset_name, csv_folder):
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


def accuracies_to_regression_vector(accuracies):
    return np.array(list(accuracies.values()))


def accuracies_to_classification_vector(accuracies, num_classes):
    acc_list = list(accuracies.values())
    max_acc = max(acc_list)
    idx = acc_list.index(max_acc)
    one_hot = np.zeros(num_classes)
    one_hot[idx] = 1
    return one_hot


if __name__ == '__main__':
    for dataset_config in dataset_configs:
        items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
        for _, _, dataset_name in items:
            accuracies = extract_accuracies(dataset_name, 'cifar10_results')
            print(accuracies_to_classification_vector(accuracies, num_classes=6))
