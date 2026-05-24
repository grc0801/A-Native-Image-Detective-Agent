import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("real_fake_model.h5" , compile=False)

# Create window
root = tk.Tk()
root.title("Real vs Fake Image Detector")
root.geometry("500x600")

# Label to show image
img_label = Label(root)
img_label.pack()

# Result label
result_label = Label(root, text="", font=("Arial", 20))
result_label.pack(pady=20)


def predict_image(path):
    img = cv2.imread(path)

    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_input)[0][0]

    if pred > 0.5:
        return "REAL", "green"
    else:
        return "FAKE", "red"


def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Display image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)

    img_label.config(image=img)
    img_label.image = img

    # Predict
    result, color = predict_image(file_path)
    result_label.config(text=result, fg=color)


# Upload button
upload_btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=20)

root.mainloop()