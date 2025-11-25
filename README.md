# 🍌 Banana Ripeness Predictor

An **AI-powered computer vision and machine learning system** that predicts **banana ripeness stages** (Green, Yellow, Brown, Mushy) and estimates **remaining shelf life**.  
Built using **TensorFlow, Keras, OpenCV, and FastAPI**, it features an **end-to-end ML pipeline** from image preprocessing and model training to **real-time cloud deployment** on **Heroku/Render**.

---

## 🚀 Features
- 🧠 **Deep Learning Model:** Multi-output CNN combining classification (ripeness) and regression (shelf life).  
- 🧮 **Computer Vision:** Image preprocessing and feature extraction using OpenCV and Pillow.  
- ⚙️ **End-to-End ML Pipeline:** Covers preprocessing → feature engineering → training → evaluation → deployment.  
- ☁️ **Cloud Deployment:** FastAPI backend integrated with Heroku/Render for real-time inference.  
- 🔄 **MLOps Ready:** Includes model optimization, version control, and reproducibility practices.  
- 📊 **Scalable Design:** Extendable for detecting freshness in other perishable items (e.g., apples, mangoes).

---

## 🧩 Tech Stack
**Languages & Frameworks:** Python, TensorFlow, Keras, FastAPI  
**Libraries:** OpenCV, NumPy, Pillow  
**Deployment:** Heroku / Render  
**Other Tools:** Git, Virtualenv, Jupyter Notebook  

---

## 🧠 Model Overview
- **Architecture:** Custom multi-output Convolutional Neural Network (CNN)  
- **Tasks:**  
  - *Classification:* Identify ripeness stage — Green, Yellow, Brown, Mushy  
  - *Regression:* Predict remaining shelf life (in days)  
- **Loss Function:** Weighted sum of categorical cross-entropy and MSE  
- **Optimization:** Adam optimizer with early stopping and learning rate scheduling  

---
## Model Weights 

The trained CNN model for banana ripeness and shelf life prediction is stored externally due to GitHub’s file size limit.
You can download it from Google Drive https://drive.google.com/file/d/1Um9r6cnV1pFTWq57Kx8UoMsWdbqbRPiD/view?usp=sharing
 and place it in the model/ directory before running the app.

## ⚙️ Installation & Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/banana-ripeness-predictor.git
cd banana-ripeness-predictor

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run FastAPI app locally
uvicorn app:app --reload

