# server.py
from flask import Flask, request, jsonify
import joblib
import zipfile
import os

app = Flask(__name__)

# مسار ملف الموديل المضغوط
ZIP_PATH = "rf_smote_rf_model.zip"
MODEL_PATH = "rf_smote_rf_model.pkl"
SCALER_PATH = "scaler.pkl"

# فك الضغط إذا الملفات غير موجودة
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")  # يفك كل الملفات في نفس المجلد

# تحميل الموديل والـ scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ✅ الواجهة الترحيبية
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
            "protocol": 6,
            "size": 512
        }
    })

# ✅ واجهة التوقع
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # تجهيز الـ features
    features = [[
        data.get("srcPort", 0),
        data.get("dstPort", 0),
        data.get("protocol", 0),
        data.get("size", 0)
    ]]

    # تحويل وتوقع
    X = scaler.transform(features)
    pred = model.predict(X)[0]

    # 0 = عادي → أخضر، 1 = خطر → أحمر
    response = {
        "label": int(pred),
        "isValid": (pred == 0),   # ✅ Boolean
        "color": "green" if pred == 0 else "red"  # ✅ String
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
