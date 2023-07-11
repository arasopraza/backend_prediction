from werkzeug.utils import secure_filename
from flask import jsonify
import os

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
upload_directory = '/Users/arasopraza/Python/be/data'

response = {
    "message": "",
}

def upload_file(file, komoditas):
    if file.filename == '':
        response["message"] = "File Tidak Boleh Kosong"
        return jsonify(response), 400
    if file and allowed_file(file.filename):
        new_file = "DataTraining" + komoditas.replace(" ", "") + ".csv"

        if os.path.exists(new_file):
            os.remove(new_file)
        
        # Save the file to the specified directory
        file.save(os.path.join(upload_directory, secure_filename(new_file)))
        
        response["message"] = "Upload File Sukses"
        return jsonify(response), 200
    else:
        response["message"] = "Format File Tidak Didukung"
        return jsonify(response), 400  
