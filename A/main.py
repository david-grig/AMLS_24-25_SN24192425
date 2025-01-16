from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np

from A.BreastMNISTCNN import BreastMNISTCNN
from A.BreastMNISTDataset import BreastMNISTDataset

data = np.load("/Users/david/PycharmProjects/AMLS_24-25_SN24192425/Datasets/breastmnist.npz")

print(data.files)

train_images = data["train_images"]
val_images = data["val_images"]
test_images = data["test_images"]
train_labels = data["train_labels"]
val_labels = data["val_labels"]
test_labels = data["test_labels"]

train_images = train_images / 255.0

print(f"Training data: {len(train_images)}, Validation data: {len(val_images)}, Testing data: {len(test_images)}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

train_loader = DataLoader(train_images, batch_size=32, shuffle=True)
val_loader = DataLoader(val_images, batch_size=32, shuffle=False)
test_loader = DataLoader(test_images, batch_size=32, shuffle=False)

dataset = BreastMNISTDataset(train_images, train_labels, transform=transform)

kf = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print(f"Fold {fold + 1}: Training data: {len(train_subset)}, Validation data: {len(val_subset)}")

model = BreastMNISTCNN()

print(model)
example_input = torch.randn(8, 1, 28, 28)
output = model(example_input)
print(f"Output shape: {output.shape}")