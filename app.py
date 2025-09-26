from flask import Flask, request, jsonify
import joblib
import os
import zipfile

app = Flask(__name__)

# 🔹 مسارات الملفات
ZIP_PATH = "model.zip"       # الملف المضغوط اللي رفعتو
EXTRACT_DIR = "model_files"  # مجلد لفك الضغط

# 🔹 فك الضغط إذا لم يتم من قبل
if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# 🔹 تحديد المسارات بعد فك الضغط
MODEL_PATH = os.path.join(EXTRACT_DIR, "model.pkl")
SCALER_PATH = os.path.join(EXTRACT_DIR, "scaler.pkl")

# 🔹 تحميل الموديل والـ scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # نفترض أن الـ features: [srcPort, dstPort, protocol_number, packet_size]
        features = [[
            data.get("srcPort", 0),
            data.get("dstPort", 0),
            data.get("protocol", 0),
            data.get("size", 0)
        ]]

        # تحويل + توقع
        X = scaler.transform(features)
        pred = model.predict(X)[0]

        return jsonify({"label": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # تشغيل السيرفر محلياً
    app.run(host="0.0.0.0", port=5000)
