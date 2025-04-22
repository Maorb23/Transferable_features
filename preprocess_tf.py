import os
import shutil
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
import argparse


# --- RemappedSubset for label remapping ---
class RemappedSubset(Dataset):
    def __init__(self, dataset, indices, label_map):
        """
        Args:
            dataset: The original dataset.
            indices: List of indices that belong to the chosen classes.
            label_map: Dictionary mapping original labels to new labels.
        """
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        # In _filter_and_remap_dataset, add:
        print(label_map)
        # Sample check
        sample_idx = indices[0]
        orig_label = dataset.targets[sample_idx]
        print(f"Sample: Original label {orig_label} -> New label {label_map[orig_label]}")
        
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data, target = self.dataset[actual_idx]
        # Remap target label
        new_target = self.label_map[target]
        return data, new_target
    
    def __len__(self):
        return len(self.indices)

# --- Data Loader ---
class TinyImageNetSplitLoader:
    def __init__(self, train_dir, val_dir_fixed, img_size=(224, 224), batch_size=64, val_ratio=0.2, seed=42):
        self.train_dir = train_dir
        self.val_dir_fixed = val_dir_fixed
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        print("Started the class")
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]) # Like in Alexnet

        self.full_train = ImageFolder(self.train_dir, transform=self.transform)
        self.test_data = ImageFolder(self.val_dir_fixed, transform=self.transform)

        self.all_classes = np.unique(self.full_train.targets)

    def split_classes(self, num_classes_a=60):
        print("Started splitting")
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Randomly choose classes for set A and let B be the remainder
        self.classes_a = np.random.choice(self.all_classes, size=num_classes_a, replace=False)
        self.classes_b = np.setdiff1d(self.all_classes, self.classes_a)
        #print(f"✅ Split classes: {len(self.classes_a)} for A, {len(self.classes_b)} for B.")

    def _filter_and_remap_dataset(self, dataset, chosen_classes):
        # Create a mapping from original label to new label (0-indexed)
        sorted_classes = sorted(chosen_classes)
        label_map = {orig_label: new_label for new_label, orig_label in enumerate(sorted_classes)}
        # Filter indices with targets in chosen_classes
        targets = np.array(dataset.targets)
        indices = [i for i, t in enumerate(targets) if t in chosen_classes]
        # Wrap in RemappedSubset so labels are remapped
        return RemappedSubset(dataset, indices, label_map)

    def _split_train_val(self, dataset):
        val_size = int(len(dataset) * self.val_ratio)
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))

    def _make_dataloaders(self, train_set, val_set, test_set):
        return {
            "train": DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1, pin_memory=True),
            "val": DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True),
            "test": DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True),
        }

    def create_dataloaders(self):
        # Filter and remap dataset for set A and set B
        full_train_a = self._filter_and_remap_dataset(self.full_train, self.classes_a)
        full_train_b = self._filter_and_remap_dataset(self.full_train, self.classes_b)
        test_a = self._filter_and_remap_dataset(self.test_data, self.classes_a)
        test_b = self._filter_and_remap_dataset(self.test_data, self.classes_b)

        # Split into train and validation sets
        train_a, val_a = self._split_train_val(full_train_a)
        train_b, val_b = self._split_train_val(full_train_b)

        # Create dataloaders
        self.dataloaders_a = self._make_dataloaders(train_a, val_a, test_a)
        self.dataloaders_b = self._make_dataloaders(train_b, val_b, test_b)

        print("🚀 Created dataloaders for A and B.")
        return self.dataloaders_a, self.dataloaders_b, self.classes_a, self.classes_b

    def summary(self):
        print(f"🔹 Full dataset size: {len(self.full_train)} images")
        print(f"🔹 Test dataset size: {len(self.test_data)} images")
        if hasattr(self, 'dataloaders_a'):
            print(f"🧩 A set - Train: {len(self.dataloaders_a['train'].dataset)}, "
                  f"Val: {len(self.dataloaders_a['val'].dataset)}, "
                  f"Test: {len(self.dataloaders_a['test'].dataset)}")
        if hasattr(self, 'dataloaders_b'):
            print(f"🧩 B set - Train: {len(self.dataloaders_b['train'].dataset)}, "
                  f"Val: {len(self.dataloaders_b['val'].dataset)}, "
                  f"Test: {len(self.dataloaders_b['test'].dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--train_dir", type=str, default="tiny-imagenet-200/train", help="Path to the training data directory")
    parser.add_argument("--val_dir_fixed", type=str, default="tiny-imagenet-200/val_fixed", help="Path to the validation data directory")
    parser.add_argument("--num_classes_a", type=int, default=100, help="Number of classes for set A")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    args = parser.parse_args()

    loader = TinyImageNetSplitLoader(
        train_dir=args.train_dir,
        val_dir_fixed=args.val_dir_fixed,
        img_size=(224, 224),
        batch_size=64,
        val_ratio=0.2,
        seed=args.seed
    )
    loader.split_classes(num_classes_a=args.num_classes_a)
    dataloaders_a, dataloaders_b, classes_a, classes_b = loader.create_dataloaders()    
    loader.summary()
    #save dataloaders a and b to a file
    torch.save(dataloaders_a, "tiny-imagenet-200/processed/dataloaders_a.pth")
    torch.save(dataloaders_b, "tiny-imagenet-200/processed/dataloaders_b.pth")
    
