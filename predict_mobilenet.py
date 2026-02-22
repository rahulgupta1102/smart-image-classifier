import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the MobileNetV2 model
model = tf.keras.models.load_model("mobilenet_classifier.h5")

# Same class order as training
classes = ['cars', 'cats', 'dogs']

# Image path
img_path = r"C:\image_classifier\dataset\test\000001.jpg"  # 👈 change if file name different

# Preprocess image
img = image.load_img(img_path, target_size=(160, 160))  # new input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
predicted_class = classes[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"\n🖼️ Image: {img_path}")
print(f"🔮 Predicted Class: {predicted_class.upper()}")
print(f"📊 Confidence: {confidence:.2f}% ✅")