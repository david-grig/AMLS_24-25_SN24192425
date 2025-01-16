from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np

from A.BreastMNISTCNN import BreastMNISTCNN
from A.BreastMNISTDataset import BreastMNISTDataset
from B.BloodMNISTCNNResNet import BloodMNISTCNNResNet
from B.BloodMNISTDataset import BloodMNISTDataset

def main():

    breast_data = np.load("/Users/david/PycharmProjects/AMLS_24-25_SN24192425/Datasets/breastmnist.npz")
    breast_train_images = breast_data["train_images"]
    breast_val_images = breast_data["val_images"]
    breast_test_images = breast_data["test_images"]
    breast_train_labels = breast_data["train_labels"]
    breast_val_labels = breast_data["val_labels"]
    breast_test_labels = breast_data["test_labels"]
    breast_model = BreastMNISTCNN()
    print("The breast model: ", breast_model)

    breast_loaders = BreastMNISTDataset.get_data_loaders(breast_train_images, breast_train_labels)

    # print(breast_loaders)

    blood_data = np.load("/Users/david/PycharmProjects/AMLS_24-25_SN24192425/Datasets/bloodmnist.npz")
    blood_train_images = blood_data["train_images"]
    blood_val_images = blood_data["val_images"]
    blood_test_images = blood_data["test_images"]
    blood_train_labels = blood_data["train_labels"]
    blood_val_labels = blood_data["val_labels"]
    blood_test_labels = blood_data["test_labels"]
    blood_model = BloodMNISTCNNResNet()
    print("The blood model: ", blood_model)

    blood_loaders = BloodMNISTDataset.get_data_loaders(blood_train_images, blood_train_labels)

    print(blood_loaders)

if __name__ == "__main__":
    main()