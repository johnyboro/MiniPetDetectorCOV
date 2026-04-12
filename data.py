import torch
import PIL.Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.pin_memory import pin_memory
from torchvision import transforms
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "list.txt"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 8

class PetDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]
        image = PIL.Image.open(IMAGES_DIR / f"{img_name}.jpg").convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label - 1  # Convert 1-37 to 0-36

def get_stats(loader):
    """Computes mean and std of the dataset."""
    print("Calculating dataset statistics...")
    sum_, res_sq_sum, nb_samples = 0, 0, 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        sum_ += data.mean(2).sum(0)
        res_sq_sum += (data**2).mean(2).sum(0)
        nb_samples += batch_samples
    
    mean = sum_ / nb_samples
    std = torch.sqrt((res_sq_sum / nb_samples) - mean**2)
    return mean, std

def get_data_loaders(val_ratio=0.1, test_ratio=0.1):
    with open(ANNOTATIONS_FILE, "r") as f:
        data = [
            (line.split()[0], int(line.split()[1]))
            for line in f if not line.startswith("#")
        ]

    remaining_data, test_data = train_test_split(
        data, test_size=test_ratio, random_state=42
    )

    adj_val_ratio = val_ratio / (1 - test_ratio)
    train_data, val_data = train_test_split(
        remaining_data, test_size=adj_val_ratio, random_state=42
    )

    base_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    
    temp_loader = DataLoader(PetDataset(train_data, base_transform), batch_size=BATCH_SIZE)
    mean, std = get_stats(temp_loader)
    print(f"Mean: {mean}, Std: {std}")

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_loader = DataLoader(
        PetDataset(train_data, train_transform), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        PetDataset(val_data, val_transform), 
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        PetDataset(test_data, val_transform),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Data ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
