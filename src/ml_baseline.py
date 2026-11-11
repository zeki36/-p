import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def extract_hog_features(data_dir="./data/dataset-resized"):
    features, labels = [], []
    class_names = os.listdir(data_dir)
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64)) # ML modelleri için daha küçük boyut
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # HOG Özellik Çıkarımı
                fd = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                features.append(fd)
                labels.append(label_idx)
                
    return np.array(features), np.array(labels), class_names

def train_svm_baseline():
    print("HOG Özellikleri Çıkarılıyor...")
    X, y, class_names = extract_hog_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("SVM Modeli Eğitiliyor (Baseline)...")
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"SVM Baseline Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

if __name__ == "__main__":
    train_svm_baseline()
