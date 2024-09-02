import os
import shutil
import subprocess

from flask import *
from flask_dropzone import Dropzone

app = Flask(__name__)

app.secret_key="dmcconet"

basedir = os.path.abspath(os.path.dirname(__file__))

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image'
)

dropzone = Dropzone(app)

@app.route("/",  methods=['GET', 'POST'])
def index():

    dirs = ["./uploads/input_file", "./uploads/style_files", "./static/gen_images"]

    open('prompt.txt', 'w').close()

    for dir in dirs:
        shutil.rmtree(dir)
        os.makedirs(dir)
    
    return render_template('index.html')

@app.route('/input_upload', methods=['POST']) 
def input_upload(): 
    if request.method == 'POST':
        f = request.files.get('file')
        file_path = os.path.join(app.config['UPLOADED_PATH'], 'input_file', f.filename)
        f.save(file_path)
    return render_template('index.html')

@app.route('/style_upload', methods=['POST']) 
def style_upload(): 
    if request.method == 'POST':
        f = request.files.get('file')
        file_path = os.path.join(app.config['UPLOADED_PATH'], 'style_files', f.filename)
        f.save(file_path)
    return render_template('index.html')

@app.route('/prompt_upload', methods=['POST']) 
def prompt_upload(): 
    if request.method == 'POST':
        prompt = request.form['prompt']

        with open('prompt.txt', 'w+', encoding="utf-8") as prompt_txt:
            prompt_txt.seek(0)
            prompt_txt.write(prompt)
            prompt_txt.truncate()

    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        if (len(os.listdir("./uploads/input_file")) == 0):
            flash("입력 이미지를 업로드하세요.")
            return render_template('index.html')
        
        if (len(os.listdir("./uploads/style_files")) == 0):
            flash("화풍 이미지를 업로드하세요.")
            return render_template('index.html')
        
        if (os.path.getsize("prompt.txt") == 0):
            flash("프롬프트를 입력하세요.")
            return render_template('index.html')
        
        subprocess.call(["python", "sd15_ipadapter.py"])
        
        images = os.listdir("./static/gen_images")
        return render_template("image.html", images=images)

    return render_template('index.html')