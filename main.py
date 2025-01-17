from warnings import filterwarnings
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np

from A.BreastMNISTCNN import BreastMNISTCNN
from A.BreastMNISTDataset import BreastMNISTDataset
from B.BloodMNISTCNNResNet import BloodMNISTCNNResNet
from B.BloodMNISTDataset import BloodMNISTDataset


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average="weighted")
    epoch_recall = recall_score(all_labels, all_preds, average="weighted")
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1

def main():
    filterwarnings('ignore')

    breast_data = np.load("./Datasets/breastmnist.npz")
    breast_train_images = breast_data["train_images"]
    breast_val_images = breast_data["val_images"]
    breast_test_images = breast_data["test_images"]
    breast_train_labels = breast_data["train_labels"]
    breast_val_labels = breast_data["val_labels"]
    breast_test_labels = breast_data["test_labels"]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ])

    train_dataset = BreastMNISTDataset(breast_train_images, breast_train_labels, transform=train_transform)
    val_dataset = BreastMNISTDataset(breast_val_images, breast_val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    breast_model = BreastMNISTCNN().to("cpu")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(breast_model.parameters(), lr=0.0005)


    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(breast_model, train_loader, criterion, optimizer, "cpu")
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = breast_model(images).squeeze(1)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    blood_data = np.load("./Datasets/bloodmnist.npz")
    blood_train_images = blood_data["train_images"]
    blood_val_images = blood_data["val_images"]
    blood_test_images = blood_data["test_images"]
    blood_train_labels = blood_data["train_labels"]
    blood_val_labels = blood_data["val_labels"]
    blood_test_labels = blood_data["test_labels"]

    blood_train_labels = blood_train_labels.flatten()
    blood_val_labels = blood_val_labels.flatten()

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ])

    train_dataset = BloodMNISTDataset(blood_train_images, blood_train_labels, transform=train_transform)
    val_dataset = BloodMNISTDataset(blood_val_images, blood_val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BloodMNISTCNNResNet().to("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

    # early_stop = False

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, "cpu")
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, "cpu")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")


if __name__ == "__main__":
    main()