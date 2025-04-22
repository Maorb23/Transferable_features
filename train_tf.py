# transfer_experiments_corrected.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet, AlexNet_Weights # Use modern import
import numpy as np
import random
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from preprocess_tf import TinyImageNetSplitLoader, RemappedSubset
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200  # Number of classes in Tiny ImageNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    import torch.nn as nn

def create_custom_alexnet_classifier(num_classes, dropout_rate=0.5):
    """Creates the custom sequential classifier block."""
    input_features = 256 * 6 * 6
    layers = [
        nn.Dropout(p=dropout_rate),
        nn.Linear(input_features, 1024),  
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(1024, 1024), 
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),  
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),  
        ]
    return nn.Sequential(*layers)

def create_lighter_alexnet(num_classes, use_pretrained_features=True):
    weights = AlexNet_Weights.IMAGENET1K_V1 if use_pretrained_features else None
    model = alexnet(weights=weights)
    # build classifier directly on GPU (or whatever DEVICE is)
    model.classifier = create_custom_alexnet_classifier(num_classes)
    return model.to(DEVICE)


def train_and_evaluate_model(model, data_loader, dropout_rate=0.5, epochs=20, optimizer_type="ADAM", lr=0.005, patience=3, weight_decay=1e-3):
    device = DEVICE
    criterion = nn.CrossEntropyLoss()
    ### Used only ADAM, SWA maybe next
    if optimizer_type == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
        use_swa = False
        swa_model = None
    elif optimizer_type == "SWA": ### Not used in the training
        base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
        optimizer = base_optimizer
        swa_model = AveragedModel(model)
        swa_start = int(epochs * 0.4)
        cycle_length = swa_start
        scheduler = CosineAnnealingLR(optimizer, T_max=cycle_length)
        swa_scheduler = SWALR(optimizer, anneal_strategy="linear", swa_lr=0.01)
        use_swa = True
    else:
        raise ValueError("Unsupported optimizer. Choose 'ADAM' or 'SWA'.")

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    swa_val_losses, swa_val_accuracies = [], []

    print('🚀 Started Training:')
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        train_bar = tqdm(data_loader["train"], desc=f"Epoch {epoch+1} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(data_loader["train"])
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if optimizer_type == "SWA":
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                print(f"SWA Scheduler Step — epoch {epoch}")
            else:
                scheduler.step()
                print(f"Regular scheduler step — epoch {epoch}")
        
        
        model_to_eval = swa_model if (optimizer_type == "SWA" and epoch >= swa_start) else model
        model_to_eval.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        val_bar = tqdm(data_loader["val"], desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_to_eval(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(data_loader["val"])
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if optimizer_type == "ADAM":
             scheduler.step(val_loss)

        logger.warning(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if optimizer_type == "SWA" and epoch >= swa_start:
            update_bn(data_loader["train"], swa_model, device=device)
            swa_model.eval()
            swa_val_loss, swa_correct, swa_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in data_loader["val"]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = swa_model(inputs)
                    swa_val_loss += criterion(outputs, labels).item()
                    _, preds = torch.max(outputs, 1)
                    swa_correct += (preds == labels).sum().item()
                    swa_total += labels.size(0)
            swa_val_loss /= len(data_loader["val"])
            swa_val_accuracy = swa_correct / swa_total
            swa_val_losses.append(swa_val_loss)
            swa_val_accuracies.append(swa_val_accuracy)
            logger.warning(f"Epoch {epoch+1} | SWA Val Loss: {swa_val_loss:.4f}, SWA Val Acc: {swa_val_accuracy:.4f}")

    if optimizer_type == "SWA":
        update_bn(data_loader["train"], swa_model, device=device)
        final_model = swa_model
        val_accuracies = val_accuracies[:swa_start] + swa_val_accuracies
        val_losses = val_losses[:swa_start] + swa_val_losses
    else:
        final_model = model

    return final_model, train_losses, train_accuracies, val_losses, val_accuracies

# --- BnB: Copy N layers ---
def create_bnb_model(num_layers_to_copy, num_classes):
    """
    Creates BnB model by copying first N layers from base_model_B,
    and initializing remaining layers randomly
    """
    pretrained = create_lighter_alexnet(num_classes, use_pretrained_features=True)
    model = create_lighter_alexnet(num_classes, use_pretrained_features=False)
    
    # Copy first N layers from base_model_B (0-indexed)
    for i in range(num_layers_to_copy):  # Correct: 0 to num_layers_to_copy-1
        model.features[i].load_state_dict(pretrained.features[i].state_dict())
        logger.warning(f"Copied layer {i} from base B")
    
    # Randomize remaining layers (num_layers_to_copy onward)
    for i in range(num_layers_to_copy, len(model.features)):
        for param in model.features[i].parameters():
            if param.dim() > 1:  # Weights
                nn.init.kaiming_normal_(param)

            else: 
                nn.init.constant_(param, 0)
    
    return model.to(DEVICE)

def create_anb_model(base_model_A, num_layers_to_freeze, num_classes):
    """
    Creates AnB model by freezing first N layers from base_model_A,
    and using base_model_B's weights for remaining layers
    """
    model = create_lighter_alexnet(num_classes, use_pretrained_features=False)
    
    # Freeze first N layers from A (0 to num_layers_to_freeze-1)
    for i in range(num_layers_to_freeze):
        model.features[i].load_state_dict(base_model_A.features[i].state_dict())
        for param in model.features[i].parameters():
            param.requires_grad = False
        logger.warning(f"Froze layer {i} from base A")
    
    # Randomize the weights of the remaining layers
    for i in range(num_layers_to_freeze, len(model.features)):
        #model.features[i].load_state_dict(base_model_B.features[i].state_dict())
        for param in model.features[i].parameters():
            if param.dim() > 1:  # Weights
                nn.init.kaiming_normal_(param)
    
    return model.to(DEVICE)

def fine_tune_model(model, data_loader, lr = 0.0001,epochs = 10):
    """
    Fine-tune the model:
    - Unfreeze all layers
    - Train with a lower learning rate
    """
    # --- Unfreeze all layers ---
    for param in model.parameters():
        param.requires_grad = True

    # --- Then fine-tune with small learning rate ---
    final_model, train_losses, train_accuracies, val_losses, val_accuracies = train_and_evaluate_model(
        model,
        data_loader,
        dropout_rate=0.5,
        epochs=epochs,
        optimizer_type="ADAM",
        lr=lr,   # Small learning rate for fine-tuning
        patience=3,
        weight_decay=1e-3
    )
    return final_model, train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate AlexNet on Tiny ImageNet")
    parser.add_argument("--dataloader_a", type=str, default="tiny-imagenet-200/processed/dataloaders_a.pth", help="Path to the training data loader")
    parser.add_argument("--data_loader_b", type=str, default="tiny-imagenet-200/processed/dataloaders_b.pth", help="Path to the validation data loader")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before early stopping")   
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--optimizer_type", type=str, default="ADAM", help="Optimizer type for training")
    parser.add_argument("--idx", type=int, default=1, help="Index for saving results")

    args = parser.parse_args()
    dataloaders_a = torch.load(args.dataloader_a, weights_only=False)
    dataloaders_b = torch.load(args.data_loader_b, weights_only=False)

    # --- Create base models ---
    base_A = create_lighter_alexnet(num_classes=100, use_pretrained_features=True)
    base_B = create_lighter_alexnet(num_classes=100, use_pretrained_features=True)

    # 4. Train base models
    print("\nTraining base model B...")
    base_B, train_losses_B, train_acc_B, val_losses_B, val_acc_B = train_and_evaluate_model(
        model=base_B,
        data_loader=dataloaders_b,
        dropout_rate=0.5,
        epochs=10,
        optimizer_type="ADAM",
        lr=0.0001,
        patience=3,
        weight_decay=1e-3
    )

    print("\nTraining base model A...")
    base_A, train_losses_A, train_acc_A, val_losses_A, val_acc_A = train_and_evaluate_model(
        model=base_A,
        data_loader=dataloaders_a,
        dropout_rate=0.5,
        epochs=10,
        optimizer_type="ADAM",
        lr=0.0001,
        patience=3,
        weight_decay=1e-3
    )
    # --- Save base model results ---
    base_model_results = {
        "base_A": {
            "train_losses": train_losses_A,
            "train_accuracies": train_acc_A,
            "val_losses": val_losses_A,
            "val_accuracies": val_acc_A
        },
        "base_B": {
            "train_losses": train_losses_B,
            "train_accuracies": train_acc_B,
            "val_losses": val_losses_B,
            "val_accuracies": val_acc_B
        }
    }
    torch.save(base_model_results, f"results/base_model_results_{args.idx}.pth")

    # 5. Create BnB models
    bnb_results = {}   # Before fine-tuning (BnB)
    bnb_plus_results = {}  # After fine-tuning (BnB+)

    bnb_models = [create_bnb_model(base_B, i, 100) for i in range(1, 6)]

    for idx, model in enumerate(bnb_models, 1):
        print(f"\n🏋️‍♂️ Training B{idx}B model (frozen {idx} layers)...")

        # --- First Training (BnB) ---
        model = model.to(DEVICE)
        model, train_losses, train_acc, val_losses, val_acc = train_and_evaluate_model(
            model,
            dataloaders_b,
            dropout_rate=0.5,
            epochs=6,
            optimizer_type="ADAM",
            lr=0.0001, 
            patience=3,
            weight_decay=1e-3
        )

        # --- Save Initial Training Results ---
        bnb_results[f"B{idx}B"] = {
            "model": model,  # (optional)
            "train_losses": train_losses,
            "train_accuracies": train_acc,
            "val_losses": val_losses,
            "val_accuracies": val_acc,
        }
        torch.save(bnb_results, "results/bnb_results_{args.idx}.pth")

        print(f"✅ Saved B{idx}B results!")

        # --- Fine-Tuning (BnB+) ---
        print(f"\n🎯 Fine-tuning B{idx}B model...")
        model, train_losses_ft, train_acc_ft, val_losses_ft, val_acc_ft = fine_tune_model(model, dataloaders_b, epochs = 5)

        # --- Save Fine-Tuning Results ---
        bnb_plus_results[f"B{idx}B+"] = {
            "model": model,  # (optional)
            "train_losses": train_losses_ft,
            "train_accuracies": train_acc_ft,
            "val_losses": val_losses_ft,
            "val_accuracies": val_acc_ft,
        }
        torch.save(bnb_plus_results, "results/bnb_plus_results_{args.idx}.pth")

        print(f"✅ Saved B{idx}B+ results!")

    anb_results = {}   # Before fine-tuning (BnB)
    anb_plus_results = {}  # After fine-tuning (BnB+)

    # 7. Create AnB models
    print("\nCreating AnB models...")
    anb_models = [create_anb_model(base_A, base_B, i, 100) for i in range(1, 6)]


    for idx, model in enumerate(anb_models, 1):
        print(f"\n🏋️‍♂️ Training A{idx}B model (frozen {idx} layers)...")

        # --- First Training (BnB) ---
        model = model.to(DEVICE)
        model, train_losses, train_acc, val_losses, val_acc = train_and_evaluate_model(
            model,
            dataloaders_b,
            dropout_rate=0.5,
            epochs=6,
            optimizer_type="ADAM",
            lr=0.0001, 
            patience=3,
            weight_decay=1e-3
        )

        # --- Save Initial Training Results ---
        anb_results[f"A{idx}B"] = {
            "model": model,  # (optional)
            "train_losses": train_losses,
            "train_accuracies": train_acc,
            "val_losses": val_losses,
            "val_accuracies": val_acc,
        }
        torch.save(anb_results, "results/anb_results_{args.idx}.pth")

        print(f"✅ Saved A{idx}B results!")

        # --- Fine-Tuning (BnB+) ---
        print(f"\n🎯 Fine-tuning A{idx}B model...")
        model, train_losses_ft, train_acc_ft, val_losses_ft, val_acc_ft = fine_tune_model(model, dataloaders_b, epochs = 5)

        # --- Save Fine-Tuning Results ---
        anb_plus_results[f"A{idx}B+"] = {
            "model": model,  # (optional)
            "train_losses": train_losses_ft,
            "train_accuracies": train_acc_ft,
            "val_losses": val_losses_ft,
            "val_accuracies": val_acc_ft,
        }
        torch.save(anb_plus_results, "results/anb_plus_results_{args.idx}.pth")
        print(f"✅ Saved B{idx}B+ results!")
