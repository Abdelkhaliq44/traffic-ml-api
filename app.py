# server.py
from flask import Flask, request, jsonify
import joblib
import zipfile
import os

app = Flask(__name__)

# ===============================
# 1. أسماء الملفات
# ===============================
ZIP_PATH = "rf_model_4features.zip"    # اسم ملف الـ zip
MODEL_PATH = "rf_model_4features.pkl"  # الموديل
SCALER_PATH = "scaler_4features.pkl"   # السكالر
PROTO_ENCODER_PATH = "protocol_encodr.pkl"  # ← الاسم الجديد
TARGET_ENCODER_PATH = "target_encoder.pkl"

# ===============================
# 2. فك الضغط (إذا لازم)
# ===============================
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

# ===============================
# 3. تحميل الملفات
# ===============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_protocol = joblib.load(PROTO_ENCODER_PATH)
le_target = joblib.load(TARGET_ENCODER_PATH)

# ===============================
# 4. الواجهة الترحيبية
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚦 Traffic ML API is running!",
        "endpoints": {
            "/": "واجهة ترحيبية",
            "/predict": "إرسال بيانات الشبكة (srcPort, dstPort, protocol, size) للحصول على التوقع"
        },
        "example": {
            "srcPort": 12345,
            "dstPort": 80,
            "protocol": "tcp",  # ← نص (سيتم ترميزه)
            "size": 512
        }
    })

# ===============================
# 5. واجهة التوقع
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ترميز البروتوكول
    protocol_val = le_protocol.transform([str(data.get("protocol", "tcp"))])[0]

    # تجهيز الـ features (4 فقط)
    features = [[
        data.get("srcPort", 0),
        data.get("dstPort", 0),
        protocol_val,
        data.get("size", 0)
    ]]

    # Scaling
    features_scaled = scaler.transform(features)

    # توقع
    pred = model.predict(features_scaled)[0]
    label = le_target.inverse_transform([pred])[0]

    response = {
        "label": label,                     # الاسم الحقيقي
        "isValid": (label == "normal"),     # إذا كان طبيعي
        "color": "green" if label == "normal" else "red"
    }
    return jsonify(response)

# ===============================
# 6. تشغيل السيرفر
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
