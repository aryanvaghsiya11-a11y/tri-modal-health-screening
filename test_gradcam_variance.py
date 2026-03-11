import os
import requests
import json
import time
import numpy as np
import cv2

url = 'http://127.0.0.1:5000/api/predict'
base_dir = r'd:\projects\project health'
samples_dir = os.path.join(base_dir, 'samples')
static_dir = os.path.join(base_dir, 'static')

if not os.path.exists(samples_dir):
    print(f"Error: Directory '{samples_dir}' not found.")
    exit(1)

images = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3] # Test first 3

print(f"Testing {len(images)} images for Grad-CAM visual variance...")
print("-" * 50)

results = []

for image_name in images:
    file_path = os.path.join(samples_dir, image_name)
    print(f"Uploading {image_name}...")
    
    with open(file_path, 'rb') as f:
        try:
            r = requests.post(
                url, 
                files={'chest_xray': f}, 
                data={'age': '50', 'gender': 'Male', 'heart_rate': '80', 'spo2': '95'}
            )
            data = r.json()
            if data.get('status') == 'success':
                # Grad-CAM URL example: /static/gradcam_1773213368.jpg
                gradcam_url = data.get('gradcam_url')
                if gradcam_url:
                    gradcam_filename = gradcam_url.split('/')[-1]
                    gradcam_path = os.path.join(static_dir, gradcam_filename)
                    
                    if os.path.exists(gradcam_path):
                        # Read the heatmap image
                        gc_img = cv2.imread(gradcam_path)
                        mean_val = np.mean(gc_img)
                        results.append({
                            'image': image_name,
                            'gradcam_file': gradcam_filename,
                            'mean_brightness': mean_val
                        })
                        print(f"  -> Grad-CAM Generated: {gradcam_filename} (Mean Brightness: {mean_val:.2f})")
                    else:
                        print(f"  -> Error: Grad-CAM file not found at {gradcam_path}")
                else:
                    # Older API response might use different key
                    print(f"  -> Error: No gradcam_url in response keys: {list(data.keys())}")
            else:
                print(f"  -> Error: {data.get('error')}")
        except Exception as e:
            print(f"  -> Exception: {e}")
    print("-" * 50)
    time.sleep(1)

# Verification Logic
if len(results) >= 2:
    diffs = []
    for i in range(len(results)-1):
        d = abs(results[i]['mean_brightness'] - results[i+1]['mean_brightness'])
        diffs.append(d)
    
    avg_diff = sum(diffs) / len(diffs)
    print(f"\nFinal Variance Check: Average Brightness Difference between Grad-CAMs: {avg_diff:.4f}")
    if avg_diff > 0.01:
        print("SUCCESS: Visual heatmaps are dynamically changing per image!")
    else:
        print("FAILURE: Heatmaps appear identical.")
else:
    print("Not enough results to compare.")
