import cv2
import numpy as np
import requests
import json

url = 'http://127.0.0.1:5000/api/predict'
file_path = 'static/gradcam_placeholder.jpg'

print(f"Testing upload with {file_path}...")
with open(file_path, 'rb') as f:
    r = requests.post(
        url, 
        files={'chest_xray': f}, 
        data={'age': '45', 'gender': 'Male', 'heart_rate': '80', 'spo2': '95'}
    )
    print("Response Code:", r.status_code)
    try:
        print("Response JSON:", json.dumps(r.json(), indent=2))
    except:
        print("Raw Output:", r.text)
