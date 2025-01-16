from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

class BreastMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def compute_mean_std(images):
        mean = np.mean(images, axis=(0, 1, 2))
        std = np.std(images, axis=(0, 1, 2))
        return mean, std

    def get_data_loaders(train_images, train_labels):
        mean, std = BreastMNISTDataset.compute_mean_std(train_images)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToTensor()
        ])

        dataset = BreastMNISTDataset(train_images, train_labels, transform=transform)
        loaders = []

        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            loaders.append((train_loader, val_loader))

        return loaders


