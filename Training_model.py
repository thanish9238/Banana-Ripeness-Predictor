import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# =========================
# 1. Load CSV & Split Data
# =========================
df = pd.read_csv("banana_labels.csv")

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['class_id']
)

# =========================
# 2. Data Pipeline
# =========================
IMG_SIZE = (224, 224)

def load_image(img_path, class_id, days_left):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, {
        "class_output": tf.one_hot(class_id, 4),
        "reg_output": tf.cast(days_left, tf.float32)
    }

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df['image_path'], train_df['class_id'], train_df['days_left'])
).map(load_image).shuffle(500).batch(32)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df['image_path'], val_df['class_id'], val_df['days_left'])
).map(load_image).batch(32)

# =========================
# 3. Build Multi-output Model
# =========================
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(224,224,3)
)
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)

# Classification head
class_output = layers.Dense(4, activation="softmax", name="class_output")(x)
# Regression head
reg_output = layers.Dense(1, activation="linear", name="reg_output")(x)

model = models.Model(inputs=base_model.input, outputs=[class_output, reg_output])

model.compile(
    optimizer="adam",
    loss={"class_output": "categorical_crossentropy", "reg_output": "mse"},
    metrics={"class_output": "accuracy", "reg_output": "mae"}
)

model.summary()

# =========================
# 4. Train Model
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# =========================
# 5. Save Model as .h5
# =========================
model.save("banana_model.h5")
print("✅ Model saved as banana_model.h5")

# =========================
# 6. Predict any banana image selected by user
# =========================
id_to_category = {0: "Green", 1: "Yellow", 2: "Brown Spots", 3: "Black/Mushy"}

# Load saved model
model = tf.keras.models.load_model("banana_model.h5")

# Prompt user to select any image
Tk().withdraw()  # Hide the root window
img_path = askopenfilename(title="Select a banana image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not img_path:
    print("❌ No image selected. Exiting.")
    exit()

# Load and preprocess the selected image
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"❌ Could not read image: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, IMG_SIZE) / 255.0
img_resized = np.expand_dims(img_resized, axis=0)

# Predict
pred_class, pred_days = model.predict(img_resized)
pred_class_id = np.argmax(pred_class[0])
pred_category = id_to_category[pred_class_id]
pred_days_left = round(float(pred_days[0][0]))

print(f"✅ Selected Image Prediction: {pred_category}, {pred_days_left} days left")
