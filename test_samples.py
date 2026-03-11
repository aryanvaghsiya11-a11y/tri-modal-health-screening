import os
import requests
import json
import time

url = 'http://127.0.0.1:5000/api/predict'
base_dir = r'd:\projects\project health'
samples_dir = os.path.join(base_dir, 'samples')

if not os.path.exists(samples_dir):
    print(f"Error: Directory '{samples_dir}' not found.")
    exit(1)

images = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(images)} images in {samples_dir}. Testing API for health diversity and SpO2=100 limit...")
print("-" * 70)

stats = {"Normal/Low Risk": 0, "Infiltration Detected": 0}

for image_name in images:
    file_path = os.path.join(samples_dir, image_name)
    print(f"Testing {image_name}...")
    with open(file_path, 'rb') as f:
        # Testing with SpO2=100 as requested
        try:
            r = requests.post(
                url, 
                files={'chest_xray': f}, 
                data={'age': '45', 'gender': 'Male', 'heart_rate': '75', 'spo2': '100'}
            )
            data = r.json()
            if data.get('status') == 'success':
                score = data.get('confidence_score_overall')
                pred = data.get('prediction')
                ens = data.get('ensemble_breakdown', {})
                stats[pred] = stats.get(pred, 0) + 1
                
                print(f"  -> Prediction: {pred} ({score}%)")
                print(f"  -> Ensemble: XGB={ens.get('xgboost_tabular')}%, CNN={ens.get('neural_network_vision')}%, Meta={ens.get('meta_learner_fusion')}%")
            else:
                print(f"  -> Error: {data.get('error')}")
        except Exception as e:
            print(f"  -> Exception: {e}")
        
    print("-" * 70)
    time.sleep(0.2)

print("\nFinal Diversity Report:")
print(f"  - Normal/Low Risk: {stats['Normal/Low Risk']}")
print(f"  - Infiltration Detected: {stats['Infiltration Detected']}")
print("\nBulk Testing Complete.")
