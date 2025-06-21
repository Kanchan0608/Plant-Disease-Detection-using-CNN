import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from matplotlib import pyplot as plt
from plant_disease_model import *

def save_learning_curves(train_losses, valid_losses, valid_accuracies, filename="learning_curves.png"):
    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(valid_losses, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(valid_accuracies, label='Validation Accuracy')
    axs[1].set_title('Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[INFO] Learning curves saved as {filename}")

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses, valid_losses, valid_accuracies = [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_loss, val_acc = evaluate_model(model, valid_loader, criterion, device)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    save_learning_curves(train_losses, valid_losses, valid_accuracies)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_total / len(data_loader), (correct / total) * 100

def main():
    train_loader, valid_loader, test_loader, num_classes = prepare_data("PlantVillage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 40, device)
    torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()
