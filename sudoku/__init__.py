from flask import Flask

UPLOAD_FOLDER = 'D:\Programing\PYTHON\Web\SudokuAutoSolveWeb\sudoku\static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ca8b4ef9bbd818cbc2d7f072'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

from sudoku import routes