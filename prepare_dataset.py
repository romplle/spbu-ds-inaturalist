import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

classes = [
    'Limenitis_arthemis',
    'Vanessa_cardui',
    'Aglais_io',
    'Vanessa_atalanta',
    'Nymphalis_antiopa'
]

data_dir = 'butterfly_dataset'
os.makedirs(data_dir, exist_ok=True)

# Глобальная сессия для переиспользования соединений
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
    csv_path = f'{cls}/{cls}.csv'
    df = pd.read_csv(csv_path)
    
    class_dir = os.path.join(data_dir, cls)
    os.makedirs(class_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        url = row['image_url']
        img_id = row['id']
        img_path = os.path.join(class_dir, f'{img_id}.jpg')
        tasks.append((url, img_path, img_id))

# Параллельная загрузка (20–50 потоков — подберите под свой интернет/CPU)
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(download_image, task) for task in tasks]
    for future in as_completed(futures):
        print(future.result())

print('Загрузка завершена!')