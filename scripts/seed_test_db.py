import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Khởi tạo (Sử dụng Service Account lấy từ Firebase Project Settings)
cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def seed_compounds():
    df = pd.read_csv('test_data/toxic_compounds.csv')
    for index, row in df.iterrows():
        # Đẩy dữ liệu lên collection 'molecules'
        db.collection('molecules').add({
            'smiles': row['smiles'],
            'status': 'testing',
            'type': 'toxic'
        })
    print("Seed xong dữ liệu kiểm thử!")

seed_compounds()