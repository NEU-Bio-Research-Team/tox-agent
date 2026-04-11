import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# 1. Cấu hình đường dẫn môi trường
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

def initialize_firestore():
    """Kết nối tới Firebase bằng Service Account"""
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        print(f"Lỗi: Không tìm thấy file {SERVICE_ACCOUNT_PATH}")
        print("Mẹo: Tải file .json từ Firebase Console > Project Settings > Service Accounts.")
        sys.exit(1)

    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    return firestore.client()

from firebase_admin import auth

def setup_database(db):
    email = "teddy@toxagent.ai"
    
    # TỰ ĐỘNG LẤY UID TỪ AUTH
    try:
        user = auth.get_user_by_email(email)
        target_uid = user.uid
    except:
        # Nếu chưa có user trên Auth thì tạo luôn
        user = auth.create_user(email=email, password="password123")
        target_uid = user.uid
    
    print(f"Sử dụng UID: {target_uid}")
    
    user_ref = db.collection("users").document(target_uid)
    user_ref.set({
        "uid": target_uid,
        "username": "teddy_tox",
        "email": "teddy@toxagent.ai",
        "role": "researcher",
        "status": "active",
        "last_login": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    # --- BƯỚC 2: TẠO MOLECULE (Sử dụng Auto-ID mặc định) ---
    print("2. Khởi tạo Molecule với Auto-ID...")
    # Dùng .add() để Firestore tự tạo ID ngẫu nhiên, tránh trùng lặp phân tử
    _, mol_ref = db.collection("molecules").add({
        "common_name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "molecular_weight": 194.19,
        "formula": "C8H10N4O2",
        "created_by": target_uid,
        "created_at": firestore.SERVER_TIMESTAMP
    })
    generated_mol_id = mol_ref.id
    print(f"Đã tạo Molecule ID: {generated_mol_id}")

    # --- BƯỚC 3: TẠO PREDICTION (Lưu trữ kết quả Agent chi tiết) ---
    print("3. Khởi tạo Prediction liên kết dữ liệu...")
    # Một phân tử có thể có nhiều dự đoán (từ các phiên bản AI khác nhau)
    pred_ref = db.collection("predictions").document() # Auto-ID
    pred_ref.set({
        "molecule_id": generated_mol_id,    # Link tới molecules
        "user_id": target_uid,             # Link tới users (người thực hiện)
        "model_metadata": {
            "name": "Tox-Agent-DeepGNN",
            "version": "v2.5",
            "environment": "production"
        },
        "results": {
            "toxicity_score": 0.05,        # 0.0 là an toàn, 1.0 là cực độc
            "is_toxic": False,
            "prediction_label": "Non-Toxic",
            "confidence_interval": 0.98    # Độ tin cậy 98%
        },
        "logs": "Agent processed SMILES via RDKit and DeepChem.",
        "execution_time_sec": 0.45,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    print(f"\nHOÀN TẤT: Đã tạo xong 3 Collections trên Firestore")

def main():
    print("--------------------------------------------------")
    print("CẢNH BÁO: Thao tác này sẽ ghi dữ liệu lên CLOUD.")
    confirm = input("Bạn có chắc chắn muốn chạy không? (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            db = initialize_firestore()
            setup_database(db)
        except Exception as e:
            print(f"Lỗi hệ thống: {e}")
    else:
        print("Đã hủy bỏ.")

if __name__ == "__main__":
    main()