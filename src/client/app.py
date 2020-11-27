import os
from flask import (
        Flask,
        render_template,
        flash,
        request,
        redirect,
        url_for,
        send_from_directory
    )
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../uploaded_samples'
ALLOWED_EXTENSIONS = ['mp3','wav', 'flac', 'm4a'] #TODO: verify
#TODO: max file size config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config(['UPLOAD_FOLDER']), filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file in post request')
            return redirect(request.url)

        file = request.files['file']
        if not file.filename:
            flash('No file uploaded')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"Only files of type {ALLOWED_EXTENSIONS} are accepted.") 
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return redirect(url_for('uploaded_file', filename=filename))


if __name__ == '__main__':
    app.run(debug=True)
