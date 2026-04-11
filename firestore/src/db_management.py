import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, service_account_path):
        """Khởi tạo kết nối Firestore"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    # --- QUẢN LÝ USER ---
    def get_user_profile(self, uid):
        """Lấy thông tin người dùng dựa trên UID"""
        user_ref = self.db.collection("users").document(uid)
        doc = user_ref.get()
        return doc.to_dict() if doc.exists else None

    # --- QUẢN LÝ MOLECULES ---
    def add_molecule(self, smiles, common_name, creator_uid, weight=0):
        """
        Thêm một phân tử mới. 
        Kiểm tra nếu SMILES đã tồn tại thì trả về ID cũ, tránh trùng lặp.
        """
        # Kiểm tra trùng lặp dựa trên chuỗi SMILES
        existing = self.db.collection("molecules").where("smiles", "==", smiles).limit(1).get()
        if existing:
            print(f"⚠️ Phân tử {smiles} đã tồn tại.")
            return existing[0].id

        # Nếu chưa có thì thêm mới
        data = {
            "smiles": smiles,
            "common_name": common_name,
            "molecular_weight": weight,
            "created_by": creator_uid,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        _, doc_ref = self.db.collection("molecules").add(data)
        return doc_ref.id

    # --- QUẢN LÝ PREDICTIONS ---
    def save_prediction(self, mol_id, user_id, score, is_toxic, model_ver="v1.0"):
        """Lưu kết quả dự đoán độc tính"""
        prediction_data = {
            "molecule_id": mol_id,
            "user_id": user_id,
            "results": {
                "toxicity_score": score,
                "is_toxic": is_toxic,
                "confidence": 0.95 # Giá trị giả định hoặc tính toán từ model
            },
            "model_version": model_ver,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        _, doc_ref = self.db.collection("predictions").add(prediction_data)
        return doc_ref.id

    def get_predictions_by_user(self, uid):
        """Lấy lịch sử dự đoán của một người dùng cụ thể"""
        docs = self.db.collection("predictions").where("user_id", "==", uid).order_by("timestamp", direction=firestore.Query.DESCENDING).get()
        return [doc.to_dict() for doc in docs]

    # --- TRUY VẤN TỔNG HỢP ---
    def get_full_report(self, prediction_id):
        """Lấy báo cáo đầy đủ bao gồm thông tin phân tử và kết quả dự đoán"""
        pred_doc = self.db.collection("predictions").document(prediction_id).get()
        if not pred_doc.exists:
            return None
        
        pred_data = pred_doc.to_dict()
        mol_data = self.db.collection("molecules").document(pred_data['molecule_id']).get().to_dict()
        
        return {
            "prediction": pred_data,
            "molecule": mol_data
        }