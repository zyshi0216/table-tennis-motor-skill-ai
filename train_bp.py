"""
train_bp.py
------------
Train a BP neural network for table tennis motion recognition.
Reimplementation of the model described in:
"Design of a Motor Skill Recognition and Hierarchical Evaluation System for Table Tennis Players" (Shi et al., IEEE Sensors Journal, 2024)

Usage:
    python train_bp.py --X data/features/X_pca.npy --y data/features/y.npy --save runs/bpnn.pth
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os


# ------------------------------
# 1. Define BP Neural Network
# ------------------------------
class BPNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 2. Training Function
# ------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=200):
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                outputs = model(X_val)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y_val.numpy())

        acc = accuracy_score(y_true, y_pred)
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {epoch_loss:.4f}  Val Acc: {acc*100:.2f}%")

    return train_losses, val_accuracies


# ------------------------------
# 3. Main Function
# ------------------------------
def main(args):
    # Load data
    X = np.load(args.X)
    y = np.load(args.y).astype(np.int64)
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]
    print(f"Loaded X: {X.shape}, y: {y.shape}, classes: {num_classes}")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=128, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPNet(input_dim, num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    losses, accs = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)

    # Save model
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"✅ Model saved to {args.save}")

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(losses, label="Training Loss")
    plt.plot(accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("BP Neural Network Training Curve")
    plt.tight_layout()
    plt.savefig("runs/training_curve.png", dpi=200)
    plt.show()


# ------------------------------
# 4. Entry
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=str, required=True, help="Path to X_pca.npy")
    parser.add_argument("--y", type=str, required=True, help="Path to label.npy")
    parser.add_argument("--save", type=str, default="runs/bpnn.pth", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(args)
