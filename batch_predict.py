import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("image_classifier_model.h5")

# Classes (same order as training)
classes = ['cats', 'dogs', 'cars']

# Test folder
test_dir = r"C:\image_classifier\dataset\test"

print("\n🔍 Starting Batch Prediction...\n")

# Loop through all images
for img_name in os.listdir(test_dir):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(test_dir, img_name)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array, verbose=0)
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        print(f"🖼️ {img_name:20} → {predicted_class.upper():5} ({confidence:.2f}%)")

print("\n✅ Batch Prediction Completed!")