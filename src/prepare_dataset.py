import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image

classes = [
    'Limenitis_arthemis',
    'Nymphalis_antiopa',
    'Papilio_glaucus',
    'Vanessa_atalanta',
    'Vanessa_cardui'
]

data_dir = 'data/processed/'
os.makedirs(data_dir, exist_ok=True)

session = requests.Session()

def download_image(args):
    url, img_path, img_id = args
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(img_path)
        return f'Saved {img_path}'
    except Exception as e:
        return f'Error {img_id} ({url}): {e}'

tasks = []

for cls in classes:
    csv_path = f'data/raw/{cls}.csv'
    df = pd.read_csv(csv_path)
    
    class_dir = os.path.join(data_dir, cls)
    os.makedirs(class_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        url = row['image_url']
        img_id = row['id']
        img_path = os.path.join(class_dir, f'{img_id}.jpg')
        tasks.append((url, img_path, img_id))

# Параллельная загрузка
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(download_image, task) for task in tasks]
    for future in as_completed(futures):
        print(future.result())

print('Загрузка завершена!')