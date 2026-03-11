import cv2
import numpy as np
import requests

img2 = np.zeros((100, 100, 3), dtype=np.uint8)
img2[:, :, 2] = 255
cv2.imwrite('fake_color.jpg', img2)

img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
cv2.imwrite('fake_gray.jpg', img1)

url = 'http://127.0.0.1:5000/api/predict'

with open('fake_color.jpg', 'rb') as f:
    r = requests.post(url, files={'chest_xray': f}, data={'age': '45', 'gender': 'Male', 'heart_rate': '80', 'spo2': '95'})
    print("Color Image Response:", r.status_code, r.json())

with open('fake_gray.jpg', 'rb') as f:
    r = requests.post(url, files={'chest_xray': f}, data={'age': '45', 'gender': 'Male', 'heart_rate': '80', 'spo2': '95'})
    print("Grayscale Image Response:", r.status_code, r.json())
