import os
from flask import Flask, request, jsonify
from predictor import WastePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

app = Flask(__name__)

predictor = WastePredictor(MODEL_PATH, CLASS_NAMES_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({
            "error": "Missing file. Use multipart/form-data with key 'file'."
        }), 400

    f = request.files["file"]
    img_bytes = f.read()

    if not img_bytes:
        return jsonify({"error": "Empty file uploaded."}), 400


    top_k = request.form.get("top_k", "3")
    try:
        top_k = int(top_k)

    except ValueError:
        top_k = 3

    pred = predictor.predict(img_bytes, top_k=top_k)

    return jsonify({
        "filename": f.filename,
        "prediction": pred["top1"],
        "topk": pred["topk"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

