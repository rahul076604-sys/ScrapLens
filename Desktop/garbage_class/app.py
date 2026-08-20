from flask import Flask, render_template, request
from src.predict import predict_waste
from src.p import  process_user_image
import numpy as np
import os
import cv2
import base64 
from src.vitP import process_user_image_viT


app = Flask(__name__)


UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    file.save(filepath)

    '''result=predict_waste(filepath)
    return render_template('index.html',predicted_class=result[0],confidence_score=round(result[1],2),image_path=filepath)'''
    # output=  process_user_image(filepath)
    output=process_user_image_viT(filepath)
    if not isinstance(output, np.ndarray):
        return render_template('index.html',prediction=output[0],confidence=round(output[1],2),image_path=filepath)
    
    else:
        success, encoded_img = cv2.imencode('.jpg', output)
        base64_img = base64.b64encode(encoded_img).decode('utf-8')
        return render_template('index.html',
                            path=base64_img) 
    
if __name__ == "__main__":
    app.run(debug=True)