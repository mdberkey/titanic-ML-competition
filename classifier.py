from torch import tensor
import torch.nn as nn
import torch.optim as opt
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
    np_labels = filtered_data['Survived'].to_numpy()
    np_data = filtered_data.drop(labels='Survived', axis=1).to_numpy()

    clean_data = []
    for inputs, label in zip(np_data, np_labels):
        clean_data.append([inputs, label])
    return DataLoader(clean_data)


def create_model():
    nn_model = nn.Sequential(nn.Linear(8, 5), nn.ReLU(), nn.Linear(5, 2), nn.ReLU())
    return nn_model


def train_model(nn_model, train_loader, criterion=nn.CrossEntropyLoss(), epoch_amt=5):
    optimization = opt.SGD(nn_model.parameters(), lr=0.001, momentum=0.9)
    nn_model.train()

    for epoch in range(epoch_amt):
        accuracy = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data


def test_model():
    pass


if __name__ == "__main__":
    data_loader = clean_and_get_data_loader()
    model = create_model()
    train_model(model, data_loader)