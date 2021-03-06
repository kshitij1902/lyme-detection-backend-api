import flask
from flask import Flask, render_template , request , jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def DetectLyme(input_data):
    interpreter = tf.lite.Interpreter(model_path="../model/LymeMobileQuantZhangKoduru.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_data[0][0] > 0.6:
        status = 'Detected'
    else :
        status = 'Undetected'
        
    return status    

@app.route('/api/upload', methods=['POST'])
def uploadImage():
    json_data = request.json
    img_base64 = json_data["image"] ## byte file
    img_data = base64.b64decode(img_base64)
    nparr = np.fromstring(img_data, np.float32)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img_300 = cv2.resize(img_np, (300, 300))
    img_last = np.expand_dims(img_np, axis=0).astype('float32')
    
    status = DetectLyme(img_last)
    return jsonify({'status':status})
    

app.run(debug=True, use_reloader=False)