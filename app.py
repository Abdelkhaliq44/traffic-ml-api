# server_debug.py  (استعمل هذا مؤقتاً على Render لقراءة الـ logs)
from flask import Flask, request, jsonify
import joblib, zipfile, os, traceback

app = Flask(__name__)

ZIP_PATH = "rf_model_4features.zip"
MODEL_PATH = "rf_model_4features.pkl"
SCALER_PATH = "scaler_4features.pkl"
PROTO_ENCODER_PATH = "protocol_encodr.pkl"   # عدل الاسم حسب ملفك فعلاً
TARGET_ENCODER_PATH = "target_encoder.pkl"

if not os.path.exists(MODEL_PATH) and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_protocol = joblib.load(PROTO_ENCODER_PATH)
le_target = joblib.load(TARGET_ENCODER_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Traffic ML API (debug) is running", "predict":"POST /predict"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        print("RAW input:", data)

        # تحويل البروتوكول (مع حماية إذا لم يكن معرفاً)
        protocol_raw = str(data.get("protocol", "tcp"))
        if protocol_raw not in le_protocol.classes_:
            print("WARNING: protocol not in encoder.classes_:", protocol_raw)
            # نحاول إضافته مؤقتا (سيكون ترميز غير مألوف)
            # الخيار الأبسط: map إلى قيمة 0 أو إلى أقرب موجود — هنا نعيد خطأ واضح
            return jsonify({"error": f"Unknown protocol '{protocol_raw}'. Allowed: {list(le_protocol.classes_)}"}), 400

        protocol_val = int(le_protocol.transform([protocol_raw])[0])

        # تجهيز features: <-- تأكد أن هذا ترتيب و أسماء تتطابق مع التدريب
        features = [[
            int(data.get("srcPort", 0)),
            int(data.get("dstPort", 0)),
            protocol_val,
            float(data.get("size", 0))
        ]]
        print("Features (raw):", features)

        # Scaling (اطبع قبل وبعد)
        try:
            features_scaled = scaler.transform(features)
        except Exception as e:
            print("Scaler transform error:", e)
            traceback.print_exc()
            return jsonify({"error": "Scaler transform failed", "detail": str(e)}), 500

        print("Features (scaled):", features_scaled.tolist())

        # توقع + احتمالات
        pred_num = int(model.predict(features_scaled)[0])
        probs = None
        try:
            probs = model.predict_proba(features_scaled)[0].tolist()
        except Exception:
            probs = "predict_proba not available"

        label = le_target.inverse_transform([pred_num])[0]
        print("Pred:", pred_num, "Label:", label, "Probs:", probs)

        resp = {"label": label, "isValid": (label == "normal"), "color": "green" if label == "normal" else "red",
                "pred_num": pred_num, "probs": probs}
        return jsonify(resp)
    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": "internal", "detail": str(ex)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
