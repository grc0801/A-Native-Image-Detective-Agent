import sys

print("🐍 Python Version:", sys.version)
print("-" * 50)

try:
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
except:
    print("❌ TensorFlow not working")

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except:
    print("❌ OpenCV not working")

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except:
    print("❌ NumPy not working")

try:
    import kagglehub
    print("✅ KaggleHub working")
except:
    print("❌ KaggleHub not working")

print("-" * 50)

# simple test
try:
    import tensorflow as tf
    print("TF test:", (tf.constant([1,2]) + tf.constant([3,4])).numpy())
except:
    print("❌ TensorFlow computation failed")