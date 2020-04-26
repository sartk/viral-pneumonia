from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import zipfile
import os
import secrets
import Inference
from PIL import Image
import numpy as np
import csv

app = Flask(__name__)

@app.route('/upload')
def upload():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        code = get_random_code()
        fname = zipper(code)
        f.save(fname)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(os.path.join('uploads', code))
        batch_run(code, f.filename.split('.')[0])
    return send_from_directory(directory='results', filename='{}.csv'.format(code),as_attachment=True)

@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

def batch_run(code, fname):
    with open('results/sample.csv'.format(code), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for im in os.listdir('uploads/{}/{}/'.format(code, fname)):
            f = os.path.join('uploads/{}/{}/'.format(code, fname), im)
            vec = Inference.image_process(np.asarray(Image.open(f)))
            print(vec.shape)
            results = Inference.infer(vec)
            writer.writerow([im, results[0], results[1]])

def zipper(x):
    return secure_filename('{}.zip'.format(x))

def get_random_code():
    hex = str(secrets.token_hex())
    if os.path.exists(zipper(hex)):
        return get_random_code()
    return hex

if __name__ == '__main__':
   app.run(debug = True)
