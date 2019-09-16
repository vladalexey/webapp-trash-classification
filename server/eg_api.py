# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.append("..")

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from predict import predict, setup, createModel, get_transform

def delete_prev(path):

    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            continue

app = Flask(__name__)

app.root_path = os.path.join(os.getcwd())
# app.template_folder = os.path.join(os.getcwd(), 'api/templates')
# app.static_folder = os.path.join(os.getcwd(), 'api/static')

print(os.listdir(app.template_folder))
print(os.listdir(app.static_folder))

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('trash.html', filename='#')

def predict_img(img):

    model_dir = '../models/models_resnext101_32x8d_acc_ 0.951807 loss_ 0.18151'
    data_transform = get_transform()

    predictor, opt, epoch = setup(model_dir, createModel)
    pred, conf, preds = predict(predictor, img, data_transform, epoch)

    # return "Prediction: {} at {:g} confidence \nConf list: {}".format(pred, conf, preds)
    return pred, conf

@app.route('/upload', methods=  ['POST'])
def upload_file():

    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename

        # prepare directory for processing
        delete_prev(app.config['UPLOAD_FOLDER'])
        f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f) 


        pred, conf = predict_img(f)

        return render_template('trash.html', filename=filename, class_pred=pred, confidence=conf)
    else:

        print('No request')
        return render_template('trash.html', filename='#', class_pred='None', confidence='None')

@app.route('/<filename>')
def send_file(filename):

    print(filename)
    filename = os.path.basename(filename)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

app.run(debug=True, threaded=True)