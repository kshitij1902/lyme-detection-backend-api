import numpy as np
import tensorflow as tf
import cv2
import base64
from PIL import Image

print(tf.__version__)
print(np.__version__)
print(cv2.__version__)
image = Image.open('images/batman.jpg')

# summarize some details about the image 
print(image.format) 
print(image.size) 
print(image.mode)


img_input = np.asarray(image)
img_input = cv2.resize(img_input, (300, 300))
img_last = np.expand_dims(img_input, axis=0).astype('float32')

print(img_last.dtype)
print(img_last.shape)
print(img_last[0][0][0:3][:])


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/LymeMobileQuantZhangKoduru.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details)
#print(output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
#print(input_shape)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = img_last
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.allocate_tensors()
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)







    



