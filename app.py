import flask
from flask import Flask, render_template , request , jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = flask.Flask(__name__)
app.config["DEBUG"] = True
secret_key = "qfoSAN2DKhiGh8AXsER7cp5WS62JXy0M"
threshold = 0.6

def DetectLyme(input_data):
    interpreter = tf.lite.Interpreter(model_path="model/LymeMobileQuantZhangKoduru.tflite")
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
    
    d=dict();
    d['prob'] = str(output_data[0][0])
    
    if output_data[0][0] > threshold:
        d['status'] = 'Detected'
    else :
        d['status'] = 'Undetected'
         
    return d

@app.errorhandler(400)
def resource_not_found(e):
    return jsonify(error=str(e)), 400

@app.route('/api/upload', methods=['POST'])
def uploadImage():
    json_data = request.json
    
    if "key" in json_data.keys():
    #authenticating request
        key = json_data["key"]
    else:
        key = ""
        
    if (key == secret_key):
        img_base64 = json_data["image"] ## byte file
        img_data = base64.b64decode(img_base64)
        #nparr = np.fromstring(img_data, np.uint8)
        nparr = np.frombuffer(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np_RGB = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_300 = cv2.resize(img_np_RGB, (300, 300))
        img_last = np.expand_dims(img_300, axis=0).astype('float32')

        
        result = DetectLyme(img_last)
        return jsonify({'status':result['status'],'prob':result['prob']})
     
    else:
        return flask.abort(400, description="Resource not found")
    

        
if __name__=="__main__":
    app.run(debug=True, use_reloader=False)