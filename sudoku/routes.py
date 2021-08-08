from sudoku import app
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import base64
import io
from sudoku.sudoku_solver.auto_solve import AutoSolve

model_path = 'sudoku/models/digit_model28x28.h5'
model_size = 28
auto_solve = AutoSolve(model_path, model_size)


def encode_img_from_file(filename):
    im = Image.open(filename)
    data = io.BytesIO()
    im.save(data, 'PNG')
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data


def encode_img_from_arr(arr):
    im = Image.fromarray(arr)
    data = io.BytesIO()
    im.save(data, 'PNG')
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    encoded_org_img = encode_img_from_file('sudoku/static/test1.png')
    encoded_solved_img = encode_img_from_file('sudoku/static/solved.png')

    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            image = image.read()
            frame = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
            encoded_org_img = encode_img_from_arr(frame)
            res, _ = auto_solve.auto_solve(frame, False)
            encoded_solved_img = encode_img_from_arr(res)
            return render_template('index.html', org_img=encoded_org_img.decode('utf-8'), solved_img=encoded_solved_img.decode('utf-8'))

    return render_template('index.html', org_img=encoded_org_img.decode('utf-8'), solved_img=encoded_solved_img.decode('utf-8'))


if __name__ == '__main__':
    app.run()
