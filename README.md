# smart-image-classifier
An AI-powered image classification app built using TensorFlow and Streamlit.  It classifies images into Cars, Cats, or Dogs with top-3 predictions, confidence bar, and a modern dark UI.



# 🚀 Smart Image Classifier (Cars, Cats & Dogs)

An advanced **Machine Learning + Deep Learning** project built using **TensorFlow**, **Keras**, and **Streamlit**.  
This app can classify uploaded images into **Cars**, **Cats**, or **Dogs** with real-time confidence visualization.

---

## 🌟 Features
- 🧠 Trained using Convolutional Neural Networks (CNN)
- 🖼️ Upload any image and get instant predictions
- 📊 Confidence bar with top-3 predictions
- 🌑 Beautiful dark UI with neon styling
- ⚡ Built using **TensorFlow** + **Streamlit**

---

## 🏗️ Tech Stack
| Component | Technology |
|------------|-------------|
| Frontend | Streamlit (Python) |
| Model | TensorFlow / Keras |
| Data | Custom Image Dataset |
| Language | Python |
| IDE Used | VS Code |

---

## 📁 Project Structure
```
image_classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cars/
│   │   ├── cats/
│   │   └── dogs/
│   └── val/
│       ├── cars/
│       ├── cats/
│       └── dogs/
│
├── image_classifier_model.h5     # Trained CNN model
├── app.py                         # Streamlit web app
├── image_classifier.py            # Model training script
├── predict.py                     # Single image prediction
└── requirements.txt               # Dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/smart-image-classifier.git
cd smart-image-classifier
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
python -m streamlit run app.py
```

### 4️⃣ Upload any image & watch the AI classify it 🎯

---

## 🎯 Model Summary
- Input Size: **150x150x3**
- Architecture: **CNN (Conv2D + MaxPooling + Dense)**
- Optimizer: **Adam**
- Loss: **Categorical Crossentropy**
- Accuracy: ~90% (with small dataset)

---

## 📸 Screenshots

| Upload Page | Prediction Result |
|--------------|------------------|
| ![Upload](https://via.placeholder.com/400x250?text=Upload+Image) | ![Result](https://via.placeholder.com/400x250?text=Prediction+Result) |

---

## 👨‍💻 Author
**Rahul Gupta**  
📍 Mumbai, India  
🎓 R.A. Podar College – Data Science & Analytics  
📧 rahulguptawork11@gmail.com  

---

## ⭐ Show Your Support
If you like this project, give it a ⭐ on GitHub!  
Your star helps keep this project alive 🚀
