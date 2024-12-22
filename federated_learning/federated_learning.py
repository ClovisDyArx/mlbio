import torch

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Iterable, List, Tuple


# CLASSES
class DatasetFL(Dataset):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        self.X = torch.stack([d[0] for d in data])
        self.y = torch.tensor([d[1] for d in data])

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class ModelFL(nn.Module):
    def __init__(self, nb_channels: int, nb_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 7 * 7, nb_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.MaxPool2d((2, 2))(torch.relu(self.conv1(x)))
        x = nn.MaxPool2d((2, 2))(torch.relu(self.conv2(x)))
        x = nn.Dropout(0.2)(self.flatten(x))
        x = self.dense(x)
        return x


# METHODS
def average_model_params(models: Iterable, average_weight: List[float]) -> dict:
    models = list(models)
    avg_state_dict = {}

    for key in models[0].state_dict().keys():
        avg_state_dict[key] = torch.zeros_like(models[0].state_dict()[key])

    for i, model in enumerate(models):
        state_dict = model.state_dict()
        weight = average_weight[i]

        for key in state_dict:
            avg_state_dict[key] += state_dict[key] * weight

    return avg_state_dict


def update_model_params(model: nn.Module, weights: dict):
    model.load_state_dict(weights)


# MAIN LOOP
def main():
    dataset = datasets.MNIST("./data", transform=transforms.ToTensor(), download=True)
    rounds = 5
    epochs = 10
    batch_size = 32
    average_weight = [1 / 2, 1 / 2]
    nb_clients = len(average_weight)

    loader_client1 = DataLoader(
        DatasetFL([dataset[i] for i in range(2000)]),
        batch_size=batch_size,
        shuffle=True,
    )

    loader_client2 = DataLoader(
        DatasetFL([dataset[i] for i in range(2000, 4000)]),
        batch_size=batch_size,
        shuffle=True,
    )

    loaders = [loader_client1, loader_client2]

    loader_test = DataLoader(
        DatasetFL([dataset[i] for i in range(4000, 4500)]),
        batch_size=batch_size,
        shuffle=True,
    )

    clients = [ModelFL(nb_channels=1, nb_classes=10) for i in range(nb_clients)]
    optimizers = [SGD(model.parameters(), lr=0.01) for model in clients]
    model = ModelFL(nb_channels=1, nb_classes=10)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for _ in range(rounds):
        for epoch in range(epochs):
            for i in range(nb_clients):
                loss_sum = 0
                nb_samples = 0
                for X, y in loaders[i]:
                    optimizers[i].zero_grad()

                    output = clients[i](X)

                    loss = criterion(output, y)
                    loss.backward()

                    loss_sum += loss.item()
                    nb_samples += len(X)

                    optimizers[i].step()

                print(f"Epoch {epoch} - client {i} - loss: {loss_sum / nb_samples}")

        print("Updating parameters of main model...")
        update_model_params(
            model, average_model_params(clients, average_weight)
        )
        for m in clients:
            update_model_params(m, model.state_dict())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader_test:
            output = model(X)
            _, predicted = torch.max(output.data, 1)
            total += len(X)
            correct += (predicted == y).sum().item()

    print(f"Accuracy test set: {correct / total}")


if __name__ == "__main__":
    main()
