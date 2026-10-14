from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "digits_dataset.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

with np.load(DATA_PATH, allow_pickle=False) as z:
    images = z["images"].astype("float32")
    labels = z["labels"].astype("int64")

# Scale source pixel range 0–16 to 0–1 and add the channel dimension.
images = images / 16.0
images = images[..., np.newaxis]
indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=labels
)
x_train, x_test = images[train_idx], images[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

tf.keras.utils.set_random_seed(42)
model = models.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
]
model.fit(
    x_train, y_train,
    validation_split=0.15,
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)
loss, keras_accuracy = model.evaluate(x_test, y_test, verbose=0)
probs = model.predict(x_test, verbose=0)
preds = np.argmax(probs, axis=1)
accuracy = float(accuracy_score(y_test, preds))

model_path = MODEL_DIR / "digit_cnn.keras"
model.save(model_path)
metrics = {
    "test_accuracy": accuracy,
    "test_loss": float(loss),
    "keras_test_accuracy": float(keras_accuracy),
    "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    "test_indices": test_idx.tolist(),
    "train_samples": int(len(train_idx)),
    "test_samples": int(len(test_idx)),
    "classes": list(range(10)),
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"\nTest accuracy: {accuracy:.3%}")
print(f"Model saved: {model_path}")
print(f"Metrics saved: {MODEL_DIR / 'metrics.json'}")
