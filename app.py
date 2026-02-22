import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 🧠 Page Setup
st.set_page_config(page_title="Smart Image Classifier", page_icon="🧩", layout="centered")

# 💻 Custom CSS for Dark Theme + Neon Style
st.markdown("""
<style>
body {
  background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
  color: white;
  font-family: 'Poppins', sans-serif;
}
div[data-testid="stFileUploader"] {
  background: #1c1c1c;
  border-radius: 10px;
  padding: 20px;
  border: 1px solid #00b4d8;
}
h1, h2, h3 {
  color: #00b4d8;
  text-align: center;
}
.stProgress > div > div > div {
  background-color: #00b4d8;
}
</style>
""", unsafe_allow_html=True)

# 🚀 Title & Description
st.title("🚀 Smart Image Classifier")
st.write("Upload an image and let this AI model guess whether it's a **Car**, **Cat**, or **Dog**!")

# 🧩 Load the Trained Model
model = tf.keras.models.load_model("image_classifier_model.h5")

# 👇 Class order — same as your training folder
classes = ['cars', 'cats', 'dogs']

# 📤 Upload Section
uploaded_file = st.file_uploader("📂 Choose an image to classify...", type=["jpg", "jpeg", "png"])

# 🖼️ When Image Uploaded
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    st.image(uploaded_file, caption="🖼️ Uploaded Image", use_container_width=True)
    st.write("⚙️ Processing...")

    # 🔮 Prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = classes[np.argmax(predictions)]

    # 🎯 Display Top 3 Predictions
    st.subheader("🔮 Predictions:")
    top_indices = np.argsort(predictions[0])[::-1][:3]
    for i in top_indices:
        st.write(f"**{classes[i].capitalize()}** — {predictions[0][i]*100:.2f}%")

    # 📊 Confidence Bar
    st.progress(int(confidence))
    st.success(f"✅ Most likely: **{predicted_class.upper()}** ({confidence:.2f}% confidence)")

else:
    st.info("Please upload an image to start classification 🖼️")