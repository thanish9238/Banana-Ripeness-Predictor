# pip install tensorflow opencv-python numpy pillow twilio tk
# pip install tensorflow opencv-python numpy pillow twilio

import cv2
import time
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from twilio.rest import Client

# ===================================================================
# Part 1: Backend and Prediction Logic
# ===================================================================

# --- Load the Trained Keras Model ---
MODEL_PATH = r'C:\Users\ramad\OneDrive\Desktop\Banana_APP\Training_dataset\banana_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Successfully loaded trained model: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Function to Capture Image from Webcam ---
def capture_image(file_path="banana_image.jpg"):
    camera = cv2.VideoCapture(0) # Try 1 for USB webcam, 0 for default
    if not camera.isOpened():
        print("❌ Error: Could not open camera.")
        return None

    print("\n📸 Camera found! Position the banana and hold still...")
    print("Taking picture in 3 seconds...")
    time.sleep(3)
    
    ret, frame = camera.read()
    if ret:
        cv2.imwrite(file_path, frame)
        print(f"Image saved successfully as {file_path}")
    else:
        print("❌ Error: Could not capture image.")
        file_path = None
        
    camera.release()
    cv2.destroyAllWindows()
    return file_path

# --- Prediction Function ---
def predict_ripeness(image_path):
    if model is None: return "Model not loaded."
    
    img_size = (224, 224)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    print("🧠 Analyzing with the AI model...")
    prediction_list = model.predict(img_array)
    prediction_array = prediction_list[0]
    
    print(f"DEBUG: Raw model output (probabilities): {prediction_array}")
    predicted_class_index = np.argmax(prediction_array)
    print(f"DEBUG: Predicted class index: {predicted_class_index}")
    
    days_left = 4 - predicted_class_index # Adjust '4' to your number of classes
    return days_left

# ===================================================================
# Part 2: WhatsApp Notification Function
# ===================================================================

def send_whatsapp_message(prediction_result):
    # --- IMPORTANT: REPLACE WITH YOUR TWILIO CREDENTIALS ---
    account_sid = 'ACaed2da2bf11140dbd4f8b149c51aeb1a' # Found on your Twilio dashboard
    auth_token = '64f511495e66325b620fbf8c4a8d310c'                 # Found on your Twilio dashboard
    twilio_whatsapp_number = '+14155238886'       # Twilio's sandbox number
    your_whatsapp_number = '+919302955925'         # Your number, verified with the sandbox
    
    client = Client(account_sid, auth_token)
    
    message_body = f"🍌 Banana Ripeness Alert! 🍌\n\nPrediction: Your banana has about {prediction_result} day(s) left."

    try:
        message = client.messages.create(
                              body=message_body,
                              from_=f'whatsapp:{twilio_whatsapp_number}',
                              to=f'whatsapp:{your_whatsapp_number}'
                          )
        print(f"WhatsApp message sent successfully! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"❌ Error sending WhatsApp message: {e}")
        messagebox.showerror("WhatsApp Error", f"Could not send message. Please check your credentials and connection.\n\nError: {e}")
        return False

# ===================================================================
# Part 3: GUI Functions and Main Application
# ===================================================================

banana_icon_img = None

def run_prediction_and_update_gui():
    predict_button.config(state=tk.DISABLED, text="Predicting...")
    status_label.config(text="Capturing image...", fg="#FFD700")
    root.update_idletasks()

    image_file = capture_image()
    if image_file:
        status_label.config(text="Image captured. Analyzing...", fg="#9ACD32")
        root.update_idletasks()

        prediction = predict_ripeness(image_file)
        
        result_message = f"🍌 Prediction Result 🍌\n\nLooks like you have about {prediction} day(s) until your banana is overripe!"
        
        messagebox.showinfo("Banana Ripeness Status", result_message)
        status_label.config(text=f"Prediction: {prediction} day(s) left. Sending WhatsApp...", fg="#008000")
        root.update_idletasks()

        # --- SEND WHATSAPP MESSAGE ---
        send_whatsapp_message(prediction)
        
        status_label.config(text=f"Prediction complete. WhatsApp sent.", fg="#008000")
    else:
        messagebox.showerror("Capture Error", "Could not capture an image. Please check your webcam.")
        status_label.config(text="Error capturing image.", fg="#FF4500")

    predict_button.config(state=tk.NORMAL, text="Predict Now")


if __name__ == "__main__":
    if model is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Fatal Error", "Could not load the banana_model.h5 file. The application will close.")
        root.destroy()
    else:
        root = tk.Tk()
        root.title("Banana Ripeness Predictor 🍌")
        root.geometry("400x250")
        root.resizable(False, False)
        root.configure(bg="#FFF8DC")
        
        header_font = ("Helvetica", 14, "bold")
        body_font = ("Helvetica", 10)
        button_font = ("Helvetica", 12, "bold")

        root.grid_rowconfigure((0, 1, 2, 3), weight=1)
        root.grid_columnconfigure((0, 1), weight=1)

        header_label = tk.Label(root, text="Banana Ripeness Analyzer", font=header_font, bg="#FFDB58", fg="#8B4513", relief=tk.RAISED, bd=3, padx=10, pady=5)
        header_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        try:
            pil_img = Image.open("banana_icon.png").resize((64, 64), Image.LANCZOS)
            banana_icon_img = ImageTk.PhotoImage(pil_img)
            icon_label = tk.Label(root, image=banana_icon_img, bg="#FFF8DC")
            icon_label.grid(row=1, column=0, pady=5, padx=10, sticky="e")
        except FileNotFoundError:
            print("No 'banana_icon.png' found. Skipping icon display.")
            icon_label = None

        instruction_text = "Press 'Predict Now' to analyze your banana's ripeness!"
        if icon_label:
            instruction_label = tk.Label(root, text=instruction_text, font=body_font, bg="#FFF8DC", wraplength=180, justify=tk.LEFT)
            instruction_label.grid(row=1, column=1, pady=5, padx=10, sticky="w")
        else:
            instruction_label = tk.Label(root, text=instruction_text, font=body_font, bg="#FFF8DC", wraplength=350)
            instruction_label.grid(row=1, column=0, columnspan=2, pady=5)

        predict_button = tk.Button(root, text="Predict Now", command=run_prediction_and_update_gui, font=button_font, bg="#4CAF50", fg="white", activebackground="#66BB6A", relief=tk.RAISED, bd=5, padx=15, pady=8)
        predict_button.grid(row=2, column=0, columnspan=2, pady=15)

        status_label = tk.Label(root, text="Ready for prediction.", font=body_font, bg="#FFF8DC", fg="gray")
        status_label.grid(row=3, column=0, columnspan=2, pady=5)

        root.mainloop()