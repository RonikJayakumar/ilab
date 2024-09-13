import warnings

import torch
from flwr_datasets import FederatedDataset
from torchvision.models import AlexNet, efficientnet_b0
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


def load_custom_cifar10(data_dir: str, batch_size: int = 32):
    """Loads a CIFAR-10 style dataset from the specified directory."""

    # Define the transformations (resize and normalize)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure the size is consistent with CIFAR-10
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
    ])

    # Load the dataset from the specified directory
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    # Optionally split the trainset for validation
    val_size = int(0.1 * len(trainset))  # 10% validation
    train_size = len(trainset) - val_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train(
    net, trainloader, valloader, epochs, device: torch.device = torch.device("cpu")
):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:  # No need for batch["img"] / batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    # Compute train and validation loss/accuracy after training
    train_loss, train_acc = test(net, trainloader, device=device)
    val_loss, val_acc = test(net, valloader, device=device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results



def test(
    net, testloader, steps: int = None, device: torch.device = torch.device("cpu")
):
    """Validate the network on the test set."""
    print("Starting evaluation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):  # Tuple unpacking
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


def load_efficientnet(classes: int = 10):
    """Loads EfficientNetB0 from TorchVision."""
    efficientnet = efficientnet_b0(pretrained=True)
    # Adjust the classifier layer for the correct number of output classes
    model_classes = efficientnet.classifier[1].in_features
    if classes != model_classes:
        efficientnet.classifier[1] = torch.nn.Linear(model_classes, classes)
    return efficientnet


def load_alexnet(classes: int = 10):
    """Load AlexNet model from TorchVision."""
    alexnet = AlexNet(num_classes=classes)
    return alexnet


def get_model_params(model):
    """Returns a model's parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]