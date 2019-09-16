from flask import Flask, render_template, url_for, flash, request, redirect
from werkzeug.utils import secure_filename
from predict import predict, setup, get_transform
import os



ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMG_FOLDER = os.path.join('static', 'img')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/result", methods=['POST'])
def result():
    if request.method == 'POST':
        file = request.files['file']
        sample_inp = 'sample.jpg'
        model_dir = './models/models_resnext101_32x8d_acc_ 0.951807 loss_ 0.18151'

        data_transform = get_transform()

        predictor, opt, epoch = setup(model_dir, createModel)
        pred, conf, preds = predict(predictor, sample_inp, data_transform, epoch)
        posts = [{1:pred,2:conf,3:preds}]
    logo = os.path.join(app.config['UPLOAD_FOLDER'], 'treegif.gif')
    carousel = os.path.join(app.config['UPLOAD_FOLDER'],'gif1.gif')
    empty_img = os.path.join(app.config['UPLOAD_FOLDER'],'empty.png')
    return render_template('result.html',my_logo = logo, my_carousel = carousel, empty_image = empty_img, posts=posts)

@app.route("/", methods = ['GET', 'POST'])
def home():
    logo = os.path.join(app.config['UPLOAD_FOLDER'], 'treegif.gif')
    carousel = os.path.join(app.config['UPLOAD_FOLDER'],'gif1.gif')
    empty_img = os.path.join(app.config['UPLOAD_FOLDER'],'empty.png')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result',
                                    filename=filename))
    return render_template('index.html', my_logo = logo, my_carousel = carousel, empty_image = empty_img)



if __name__ == '__main__':
    app.run(debug=True)