import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from preprocessing import get_dataloaders
from model import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_errors_with_gradcam():
    _, _, test_loader, class_names = get_dataloaders(batch_size=1)
    model = get_resnet50(freeze_backbone=False)
    model.load_state_dict(torch.load('../models/best_resnet.pth'))
    model.to(device)
    model.eval()

    # GradCAM için hedef katman (ResNet50 son evrişim bloğu)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    correct_samples, incorrect_samples = [], []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        is_correct = (preds == labels).item()
        
        # Sadece eksik örnekler tamamlanana kadar işlemi yap
        if is_correct and len(correct_samples) < 5:
            correct_samples.append((inputs, labels, preds))
        elif not is_correct and len(incorrect_samples) < 5:
            incorrect_samples.append((inputs, labels, preds))
            
        if len(correct_samples) == 5 and len(incorrect_samples) == 5:
            break

    # Görüntüleme Fonksiyonu
    def visualize_samples(samples, title):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(title, fontsize=16)
        for i, (img_tensor, true_label, pred_label) in enumerate(samples):
            grayscale_cam = cam(input_tensor=img_tensor)[0, :]
            
            # Tensörü görselleştirilebilir RGB formata geri çevirme
            img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            axes[i].imshow(cam_image)
            axes[i].set_title(f"Gerçek: {class_names[true_label.item()]}\nTahmin: {class_names[pred_label.item()]}")
            axes[i].axis('off')
        plt.show()

    visualize_samples(correct_samples, "Doğru Sınıflandırılan Örnekler ve Grad-CAM Haritaları")
    visualize_samples(incorrect_samples, "Yanlış Sınıflandırılan Örnekler (Hata Analizi) ve Grad-CAM Haritaları")

if __name__ == "__main__":
    analyze_errors_with_gradcam()
