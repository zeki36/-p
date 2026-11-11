import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import get_dataloaders
from model import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_phase(model, train_loader, val_loader, criterion, optimizer, epochs, phase_name):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # En iyi modeli kaydet
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), '../models/best_resnet.pth')
            
    return train_losses, val_losses

def run_experiment(learning_rate, weight_decay):
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=32)
    model = get_resnet50(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 1. Aşama: Sadece FC katmanını eğitme
    print("--- 1. Aşama: Sınıflandırma Katmanı Eğitimi ---")
    optimizer1 = optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    t_loss1, v_loss1 = train_one_phase(model, train_loader, val_loader, criterion, optimizer1, epochs=10, phase_name="Aşama 1")
    
    # 2. Aşama: Son katmanları açarak ince ayar (Fine-Tuning)
    print("--- 2. Aşama: İnce Ayar (Fine-Tuning) ---")
    # layer4'ü dondurmayı bırak
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/10, weight_decay=weight_decay)
    t_loss2, v_loss2 = train_one_phase(model, train_loader, val_loader, criterion, optimizer2, epochs=10, phase_name="Aşama 2")
    
    # Grafik Çizimi (Convergence & Overfitting Analizi)
    plt.plot(t_loss1 + t_loss2, label='Eğitim Kaybı')
    plt.plot(v_loss1 + v_loss2, label='Doğrulama Kaybı')
    plt.axvline(x=10, color='r', linestyle='--', label='İnce Ayar Başlangıcı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Eğitim Analizi (LR={learning_rate}, WD={weight_decay})')
    plt.savefig('../results/loss_curve.png')
    plt.show()

if __name__ == "__main__":
    # Kontrollü Deney Çalışması
    run_experiment(learning_rate=1e-3, weight_decay=1e-4)
