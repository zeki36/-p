import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import get_dataloaders
from model import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    _, _, test_loader, class_names = get_dataloaders()
    model = get_resnet50(freeze_backbone=False)
    model.load_state_dict(torch.load('../models/best_resnet.pth'))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrik Hesaplamaları
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Test Metrikleri:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.title('Karmaşıklık Matrisi')
    plt.savefig('../results/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
