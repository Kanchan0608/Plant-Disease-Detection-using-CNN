import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

def load_images(directory_root):
    image_list, label_list = [], []
    for disease_folder in os.listdir(directory_root):
        folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.startswith("."): continue
            img_path = os.path.join(folder_path, img_name)
            if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                image_list.append(img_path)
                label_list.append(disease_folder)
    return image_list, label_list

def prepare_data(directory_root, image_size=(256, 256), batch_size=32, test_size=0.3, valid_ratio=0.5):
    from sklearn.model_selection import train_test_split
    image_paths, labels = load_images(directory_root)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("class_names.json", "w") as f:
        json.dump(list(label_encoder.classes_), f)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_encoded, test_size=test_size, stratify=labels_encoded
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=valid_ratio, stratify=temp_labels
    )
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    valid_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    with open("inference_transform.pkl", "wb") as f:
        pickle.dump(valid_test_transform, f)
    train_ds = PlantDiseaseDataset(train_paths, train_labels, train_transform)
    valid_ds = PlantDiseaseDataset(valid_paths, valid_labels, valid_test_transform)
    test_ds = PlantDiseaseDataset(test_paths, test_labels, valid_test_transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(valid_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
            len(label_encoder.classes_))
