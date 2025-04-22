import numpy as np
import cv2
import os
import joblib
from pyzbar.pyzbar import decode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load QR Code images dataset
valid_qr_path = r"C:\Users\aswat\Downloads\qr_ics\qr_dataset\real"
tampered_qr_path = r"C:\Users\aswat\Downloads\qr_ics\qr_dataset\tampered"

image_size = (200, 200)

def extract_qr_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size)
    img = img.flatten() / 255.0  
    return img

def load_dataset():
    X, y = [], []
    
    for file in os.listdir(valid_qr_path):
        X.append(extract_qr_features(os.path.join(valid_qr_path, file)))
        y.append(1)  # 1 for valid QR
    
    for file in os.listdir(tampered_qr_path):
        X.append(extract_qr_features(os.path.join(tampered_qr_path, file)))
        y.append(0)  # 0 for tampered QR
    
    return np.array(X), np.array(y)

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the model
joblib.dump(model, "qr_verification_model.pkl")
print("Model saved as qr_verification_model.pkl")
