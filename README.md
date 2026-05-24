# A-Native-Image-Detective-Agent
Built a lightweight image classifier to detect real vs fake images using a Kaggle dataset (~5000 samples/class, 100–200 subjects). Trained with a 70:30 split on MobileNetV2, then converted to TensorFlow Lite for efficient deployment in a native Android app for on-device predictions.

1. Where check_lite.py used to check the accuracy of Tensorflow Lite model
2. The check_test.py used for acquiring results via console input and output with Tensorflow Lite model
3. The convert_tflite.py used for converting the Tensorflow model into Tensorflow Lite
4. The main.py is the actual code
5. predict.py used for predicitng output with MobileNetV2
6. The train_code.py used for training the model
7. validate.py is for evaluation metric of the model

So this is being set to be developed further for deployment on Mobile Based apk integrating with more datasets,. i.e, with different classes like Objects and Live speech - detecting whether it is AI or Human.
