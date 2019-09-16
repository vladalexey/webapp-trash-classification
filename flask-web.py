from flask import Flask, render_template, url_for, flash, request, redirect
from werkzeug.utils import secure_filename
from predict import predict, setup, get_transform
import os
from predict import predict, setup, createModel, get_transform


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMG_FOLDER = os.path.join('static', 'img')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

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

def predict_img(img):

    model_dir = '../models/models_resnext101_32x8d_acc_ 0.951807 loss_ 0.18151'
    data_transform = get_transform()

    predictor, opt, epoch = setup(model_dir, createModel)
    pred, conf, preds = predict(predictor, img, data_transform, epoch)

    # return "Prediction: {} at {:g} confidence \nConf list: {}".format(pred, conf, preds)
    return pred, conf


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/result",methods=  ['POST'])
def result():
    logo = os.path.join(app.config['UPLOAD_FOLDER'], 'treegif.gif')
    carousel = os.path.join(app.config['UPLOAD_FOLDER'],'gif1.gif')
    empty_img = os.path.join(app.config['UPLOAD_FOLDER'],'empty.png')
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename

        # prepare directory for processing
        delete_prev(app.config['UPLOAD_FOLDER'])
        f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f) 


        pred, conf = predict_img(f)
        return render_template('result.html',my_logo = logo, my_carousel = carousel, empty_image = empty_img, posts=posts, filename=filename, class_pred=pred, confidence=conf)
    else:

        print('No request')
        return render_template('trash.html',my_logo = logo, my_carousel = carousel, empty_image = empty_img, filename='#', class_pred='None', confidence='None')

@app.route("/")
def home():
    logo = os.path.join(app.config['UPLOAD_FOLDER'], 'treegif.gif')
    carousel = os.path.join(app.config['UPLOAD_FOLDER'],'gif1.gif')
    empty_img = os.path.join(app.config['UPLOAD_FOLDER'],'empty.png')
    return render_template('index.html', my_logo = logo, my_carousel = carousel, empty_image = empty_img)

@app.route('/<filename>')
def send_file(filename):

    print(filename)
    filename = os.path.basename(filename)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)