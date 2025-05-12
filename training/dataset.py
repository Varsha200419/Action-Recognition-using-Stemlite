import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
class HMDBDataset(Dataset):
    def __init__(self, root_dir, clip_size=8, transform=None, frame_rate=16):
        self.root_dir = root_dir
        self.clip_size = clip_size
        self.transform = transform
        self.frame_rate = frame_rate
        self.data = self._load_data()

    def _load_data(self):
        data = []
        subfolders = sorted(os.listdir(self.root_dir))
        
        # Ensure there are 25 subfolders
        if len(subfolders) != 25:
            raise ValueError(f"Dataset must contain exactly 25 subfolders. Found {len(subfolders)}.")

        for label, action in enumerate(subfolders):
            action_path = os.path.join(self.root_dir, action)
            if not os.path.isdir(action_path):  # Skip if not a directory
                continue
            for video_folder in os.listdir(action_path):
                video_path = os.path.join(action_path, video_folder)
                if os.path.isdir(video_path):
                    all_frames = sorted(os.listdir(video_path))
                    if len(all_frames) < self.clip_size:
                        print(f"Skipping {video_path}: Fewer than {self.clip_size} frames.")
                        continue
                    data.append((video_path, label))
        return data

    def __len__(self):
        return len(self.data)


    def _load_frames(self, video_path):
        """
        Load all frames from a video folder (expected to have exactly clip_size frames).
        """
        all_frames = sorted(os.listdir(video_path))
        total_frames = len(all_frames)
        if total_frames != self.clip_size:
            raise ValueError(f"{video_path} has {total_frames} frames, expected {self.clip_size}.")
        frames = [Image.open(os.path.join(video_path, frame)) for frame in all_frames]
        return frames

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self._load_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames), label

def get_dataloader(root_dir, batch_size=8, clip_size=8, train_ratio=0.8, val_ratio = 0.1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = HMDBDataset(root_dir, clip_size=clip_size, transform=transform, frame_rate=16)
    # Get indices and labels
    indices = range(len(dataset))
    labels = [label for _, label in dataset.data]
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1-(train_ratio+val_ratio), stratify=labels, random_state=42
    )
    
    # Adjust labels for train+val split
    train_val_labels = [labels[i] for i in train_val_idx]
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio/(train_ratio+val_ratio), 
        stratify=train_val_labels, random_state=42
    )
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Log split sizes
    print(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
