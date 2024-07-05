import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg11
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Load CIFAR-10 Dataset
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return dataset

# Create Non-IID Partitions
def create_non_iid_partitions(dataset, num_clients, num_classes=10):
    data = np.array(dataset.data)
    labels = np.array(dataset.targets)

    client_data_indices = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        split = np.array_split(idx, num_clients)
        for j in range(num_clients):
            client_data_indices[j].extend(split[j])

    return client_data_indices

# Create Data Loaders for Training and Testing
def create_data_loaders(client_indices, dataset, batch_size=32, test_split=0.2, num_workers=4):
    train_loaders = []
    test_loaders = []

    for indices in client_indices:
        train_size = int((1 - test_split) * len(indices))
        test_size = len(indices) - train_size
        train_indices, test_indices = random_split(indices, [train_size, test_size])

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders

# Define VGG11 Model
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.model = vgg11(pretrained=False)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

# FedNova Logic Simulation
def fednova_update(model, local_models, local_steps, device):
    global_params = model.state_dict()
    total_steps = sum(local_steps)

    # Initialize aggregated updates
    aggregated_updates = {k: torch.zeros_like(v) for k, v in global_params.items()}

    for local_model, steps in zip(local_models, local_steps):
        local_params = local_model.state_dict()
        for k in global_params.keys():
            aggregated_updates[k] += (local_params[k] - global_params[k]) * (steps / total_steps)

    # Update global model parameters
    for k in global_params.keys():
        global_params[k] += aggregated_updates[k]

    model.load_state_dict(global_params)

# Training Function
def train(model, device, train_loader, criterion, local_epochs, config):
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    for epoch in range(local_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return len(train_loader.dataset) // train_loader.batch_size

# Training with Ray Tune
def federated_training(config, num_clients=4, batch_size=32, communication_rounds=5, local_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and Partition CIFAR-10 Dataset
    dataset = load_cifar10()
    client_indices = create_non_iid_partitions(dataset, num_clients)
    train_loaders, test_loaders = create_data_loaders(client_indices, dataset, batch_size)

    # Initialize Global Model and Loss Function
    global_model = VGG11().to(device)
    criterion = nn.CrossEntropyLoss()

    for round_num in range(1, communication_rounds + 1):
        local_models = []
        local_steps = []

        for client_id in range(num_clients):
            local_model = VGG11().to(device)
            local_model.load_state_dict(global_model.state_dict())
            steps = train(local_model, device, train_loaders[client_id], criterion, local_epochs, config)
            local_models.append(local_model)
            local_steps.append(steps)

        fednova_update(global_model, local_models, local_steps, device)

    # Evaluate the model on the validation data
    global_model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loaders[0]:  # Assuming using the first client's test data for validation
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            validation_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    tune.report(loss=validation_loss, accuracy=accuracy)

# Main Code
if __name__ == "__main__":
    ray.init()

    config = {
        "lr": tune.loguniform(1e-4, 1e-2)
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        federated_training,
        config=config,
        num_samples=10,
        scheduler=scheduler,
        resources_per_trial={"cpu": 8, "gpu": 2}
    )

    print("Best hyperparameters found were: ", analysis.best_config)
