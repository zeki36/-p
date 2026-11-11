import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import get_dataloaders
from model import get_resnet50, get_efficientnet_b0, BaselineCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_phase(model, train_loader, val_loader, criterion, optimizer, epochs, phase_name, model_name="model"):
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
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'../models/best_{model_name}.pth')
            
    return train_losses, val_losses, best_val_loss

def run_experiment_pipeline(learning_rate, weight_decay, model_type="resnet", use_augmentation=True, fine_tune=True):
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=32, use_augmentation=use_augmentation)
    criterion = nn.CrossEntropyLoss()
    
    if model_type == "resnet":
        model = get_resnet50(freeze_backbone=True).to(device)
        optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_type == "efficientnet":
        model = get_efficientnet_b0(freeze_backbone=True).to(device)
        optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_type == "baseline_cnn":
        model = BaselineCNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"\n=== EĞİTİM BAŞLIYOR: Model: {model_type} | LR: {learning_rate} | Aug: {use_augmentation} ===")
    
    # 1. Aşama Eğitim
    t_loss1, v_loss1, best_val = train_one_phase(model, train_loader, val_loader, criterion, optimizer, epochs=10, phase_name="Aşama 1", model_name=model_type)
    
    # 2. Aşama İnce Ayar (Sadece Transfer Learning modelleri için ve fine_tune açıksa)
    t_loss2, v_loss2 = [], []
    if fine_tune and model_type in ["resnet", "efficientnet"]:
        print(f"--- 2. Aşama: İnce Ayar (Fine-Tuning) {model_type} ---")
        if model_type == "resnet":
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif model_type == "efficientnet":
            for param in model.features[-2:].parameters():
                param.requires_grad = True
                
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/10, weight_decay=weight_decay)
        t_loss2, v_loss2, best_val2 = train_one_phase(model, train_loader, val_loader, criterion, optimizer2, epochs=10, phase_name="Aşama 2", model_name=f"{model_type}_finetuned")
        best_val = min(best_val, best_val2)

    return best_val

if __name__ == "__main__":
    print("\n--- GÖREV 1: DERİN ÖĞRENME MODELLERİNİN KIYASLANMASI ---")
    run_experiment_pipeline(learning_rate=1e-3, weight_decay=1e-4, model_type="baseline_cnn", fine_tune=False)
    run_experiment_pipeline(learning_rate=1e-3, weight_decay=1e-4, model_type="resnet")
    run_experiment_pipeline(learning_rate=1e-3, weight_decay=1e-4, model_type="efficientnet")

    print("\n--- GÖREV 2: KONTROLLÜ DENEY (ResNet üzerinde farklı Learning Rate testleri) ---")
    run_experiment_pipeline(learning_rate=1e-2, weight_decay=1e-4, model_type="resnet")
    run_experiment_pipeline(learning_rate=1e-4, weight_decay=1e-4, model_type="resnet")

    print("\n--- GÖREV 3: ABLASYON ÇALIŞMASI (ResNet üzerinde) ---")
    # Veri artırma (Data Augmentation) kapalı
    run_experiment_pipeline(learning_rate=1e-3, weight_decay=1e-4, model_type="resnet", use_augmentation=False)
    # Veri artırma açık ama İnce Ayar (Fine-Tuning) kapalı
    run_experiment_pipeline(learning_rate=1e-3, weight_decay=1e-4, model_type="resnet", use_augmentation=True, fine_tune=False)
