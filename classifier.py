import torch.nn as nn
import torch.optim as opt
import torch
from torch.utils.data import DataLoader
import pandas as pd


def clean_and_get_data_loader(train=True):
    filename = 'data/train.csv'
    if not train:
        filename = 'data/test.csv'

    filtered_data = pd.read_csv(filename)
    filtered_data = filtered_data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    filtered_data = filtered_data.fillna(value={'Age': 29.7, 'Embarked': 'S'})
    filtered_data = filtered_data.replace(to_replace=['male', 'female', 'S', 'C', 'Q'], value=[0, 1, 0, 1, 2])

    clean_test = []
    if not train:
        np_test = filtered_data.to_numpy()
        for inputs in np_test:
            clean_test.append(inputs)
        return DataLoader(clean_test)

    np_labels = filtered_data['Survived'].to_numpy()
    np_data = filtered_data.drop(labels='Survived', axis=1).to_numpy()

    clean_train = []
    for inputs, label in zip(np_data, np_labels):
        clean_train.append([inputs, label])
    return DataLoader(clean_train)


def create_model():
    nn_model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 2), nn.ReLU())
    return nn_model


def train_model(nn_model, train_loader, criterion=nn.CrossEntropyLoss(), epoch_amt=5):
    optimizer = opt.SGD(nn_model.parameters(), lr=0.000001, momentum=0.9)    # 0.000001 = 71% tops
    nn_model.train()

    for epoch in range(epoch_amt):
        accuracy = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, label = data
            optimizer.zero_grad()
            output = nn_model(inputs.float())
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if label.item() == torch.argmax(output, dim=1).item():
                accuracy += 1

        total = i + 1
        accuracy_perc = accuracy / total
        curr_loss = running_loss / i

        output_str = f'Train Epoch: {epoch}    Accuracy: {accuracy}/{total}({accuracy_perc:.2%}) Loss: {curr_loss:.3f}'
        print(output_str)
    print("Training Complete")


def test_model(nn_model, test_loader):
    model.eval()
    predicted_labels = []

    with torch.no_grad():
        for i, inputs in enumerate(test_loader, 892):
            output = nn_model(inputs.float())
            predicted_labels.append((i, torch.argmax(output, dim=1).item()))

    predicted_labels = pd.DataFrame(predicted_labels)
    predicted_labels = predicted_labels.set_axis(labels=["PassengerId", "Survived"], axis=1)
    predicted_labels.to_csv("data/predictions.csv", index=False)
    print("Testing Complete")


if __name__ == "__main__":
    train_loader = clean_and_get_data_loader()
    test_loader = clean_and_get_data_loader(train=False)
    model = create_model()
    train_model(model, train_loader, epoch_amt=20)
    test_model(model, test_loader)