import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="real_fake_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create UI
root = tk.Tk()
root.title("TFLite Real vs Fake Detector")
root.geometry("500x600")

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="", font=("Arial", 22))
result_label.pack(pady=20)


def predict_tflite(img_path):
    img = cv2.imread(img_path)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if output > 0.5:
        return "REAL", "green", output
    else:
        return "FAKE", "red", output


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
    result, color, confidence = predict_tflite(file_path)

    result_label.config(
        text=f"{result} ({confidence*100:.2f}%)",
        fg=color
    )


btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
btn.pack(pady=20)

root.mainloop()