import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import logging
import json
import sys
import argparse

# Add the src directory to sys.path
sys.path.append('/user/HS402/zs00774/Downloads/action-recognition-vit/src')

from models.timesformer_1 import load_timesformer_model
from training.dataset import get_dataloader

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_dir, epochs, batch_size, learning_rate):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Batch Size: {batch_size}, Learning Rate:{learning_rate}, No. of Epochs:{epochs}, Optimizer: SGD, Loss Function: Cross Entropy")
    train_loader, val_loader, test_loader = get_dataloader(data_dir, batch_size)
    processor, model = load_timesformer_model()
    model = model.to(device)  # Move model to GPU
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scaler = GradScaler()  # For mixed precision training
    writer = SummaryWriter()
    metrics = {"training_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            if batch_idx == 0:  # Log only for the first batch of each epoch
                logging.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}: Batch size: {inputs.size(0)}, Input shape: {inputs.size()}")
            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        metrics["training_loss"].append(epoch_loss)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save metrics to a JSON file
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)

    writer.close()
    torch.save(model.state_dict(), "timesformer_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    # Use the preprocessed dataset path
    parser = argparse.ArgumentParser(description="Train a model for action recognition.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    # Call the train_model function with parsed arguments
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

