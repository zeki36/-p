import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir="./data/dataset-resized", batch_size=32):
    # Eğitim için Veri Artırma (Data Augmentation) ve Normalizasyon
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Çevirme
        transforms.RandomRotation(15),     # Döndürme
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Doğrulama/Test için Sadece Yeniden Boyutlandırma ve Normalizasyon
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Veri setini yükleme ve dönüşümleri atama
    full_dataset = datasets.ImageFolder(data_dir)
    
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Tekrarlanabilirlik için random seed (42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Dataset dönüşümlerini bağımsız uygulama
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = eval_transforms
    test_dataset.dataset.transform = eval_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes