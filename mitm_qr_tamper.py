import cv2
import numpy as np
import random
import qrcode
from pyzbar.pyzbar import decode
from PIL import Image
import os

def add_ai_noise(image_path, save_path, noise_level=5):
   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    
    noisy_image = image.copy()
    for _ in range(noise_level * 100):  # Adjust number of noise points
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        noisy_image[y, x] = random.choice([0, 255])  
    
    cv2.imwrite(save_path, noisy_image)
    print(f"Tampered QR saved to {save_path}")

def tamper_qr_content(original_qr_path, tampered_qr_path):
    """
    Reads the original QR, modifies the encoded content, and saves a tampered QR.
    """
    qr_image = Image.open(original_qr_path)
    decoded_data = decode(qr_image)
    
    if decoded_data:
        qr_content = decoded_data[0].data.decode()
        
        
        tampered_content = qr_content[:-2] + "!!" if len(qr_content) > 2 else qr_content[::-1]
        
        # Generate a new tampered QR code
        tampered_qr = qrcode.make(tampered_content)
        tampered_qr.save(tampered_qr_path)
        print(f"Tampered QR content saved to {tampered_qr_path}")
    else:
        print("QR decoding failed!")


original_qr_path = r"C:\Users\aswat\Downloads\qr_ics\encrypted_qr.png"
tampered_qr_path = r"C:\Users\aswat\Downloads\qr_ics\encrypted_1qr.png"

tamper_qr_content(original_qr_path, tampered_qr_path)
add_ai_noise(tampered_qr_path, tampered_qr_path)
