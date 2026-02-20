import json
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

class WastePredictor:
    def __init__(self, model_path: str, class_names_path: str):
        
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": tf.keras.applications.resnet50.preprocess_input}
        )

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        if not isinstance(self.class_names, list) or len(self.class_names) == 0:
            raise ValueError("class_names.json must be a non-empty JSON list of class names.")

    def _prepare(self, img_bytes: bytes) -> np.ndarray:
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)

        x = tf.expand_dims(img, axis=0)
        return x.numpy()

    def predict(self, img_bytes: bytes, top_k: int = 3):
        x = self._prepare(img_bytes)
        probs = self.model.predict(x, verbose=0)[0]

        top_k = max(1, min(int(top_k), len(probs)))
        idxs = probs.argsort()[::-1][:top_k]

        topk = []
        for i in idxs:
            topk.append({
                "class": self.class_names[int(i)],
                "prob": round(float(probs[int(i)]), 6)
            })

        return {"top1": topk[0], "topk": topk}
