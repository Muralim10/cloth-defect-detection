import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('model_Thread.keras')

img_path = "C:\\Users\\SANJAI.R\\Downloads\\thread defect\\thread test\\Knot\\17.jpg"

img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array /= 255.0 


prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)

# Interpret the prediction
class_names = ['Breakage', 'Knot', 'Normal', 'Unevenness']  
predicted_class_name = class_names[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")