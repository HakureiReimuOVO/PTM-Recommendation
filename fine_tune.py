import csv
import gc
import evaluate
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import *
from model_loader import get_model_and_processor
from dataset_loader import *
from slice_dataset import get_all_datasets_and_idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# def get_learning_rate(model_name):
#     if 'resnet' in model_name or 'mobilenet' in model_name or 'mobilevit' in model_name:
#         return 1e-3
#     elif 'vit' in model_name or 'beit' in model_name or 'swin' in model_name:
#         return 1e-4
#     else:
#         return 2e-5


def train_model_on_data_loader(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-4,
                               model_name='', dataset_name='', log_writer=None, log_file=None):
    print(f'Training model {model_name} on {dataset_name}...')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    model.to(device)

    metric = evaluate.load("./metrics/accuracy")
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        # TODO: Test
        model.eval()

        total_val_loss = 0
        for batch in tqdm(test_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images).logits
                val_loss = criterion(outputs, labels).item()
                total_val_loss += val_loss
                predictions = torch.argmax(outputs, dim=-1)
                metric.add_batch(predictions=predictions, references=labels)

        avg_val_loss = total_val_loss / len(test_loader)
        eval_metric = metric.compute()
        accuracy = eval_metric['accuracy']
        print(f"Validation Accuracy: {accuracy}, Loss: {avg_val_loss}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        scheduler.step(avg_val_loss)

        if log_writer is not None:
            log_writer.writerow([model_name, epoch + 1, avg_loss, accuracy])
            if log_file is not None:
                log_file.flush()

    return best_accuracy


if __name__ == '__main__':
    best_accuracy_log_file = 'result/best_accuracies.csv'
    with open(best_accuracy_log_file, mode='w', newline='') as best_acc_file:
        best_acc_writer = csv.writer(best_acc_file)
        best_acc_writer.writerow(['Dataset', 'Model', 'Best Accuracy'])

        for dataset_config in dataset_configs:
            tuples = get_all_datasets_and_idx(dataset_config['name'])

            # Test
            for dataset, _, dataset_fullname in tuples:
                train_loader, test_loader = get_data_loader(dataset_name=dataset_fullname,
                                                            image_key=dataset_config['image_key'],
                                                            label_key=dataset_config['label_key'],
                                                            batch_size=32,
                                                            train_test_split_ratio=0.8,
                                                            test=True)

                log_file = f'train_logs/{dataset_fullname}.csv'
                with open(log_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Model', 'Epoch', 'Train Loss', 'Validation Accuracy'])

                    # Test
                    for model_config in model_configs:
                        model, _ = get_model_and_processor(model_config, num_labels=2)
                        best_accuracy = train_model_on_data_loader(model, train_loader=train_loader,
                                                                   test_loader=test_loader,
                                                                   # Test
                                                                   num_epochs=10,
                                                                   model_name=model_config,
                                                                   dataset_name=dataset_fullname,
                                                                   log_writer=writer,
                                                                   log_file=file)
                        best_acc_writer.writerow([dataset_fullname, model_config, best_accuracy])
                        best_acc_file.flush()

                        del model
                        torch.cuda.empty_cache()
                        gc.collect()