import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask import jsonify

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
upload_directory = '/data'

response = {
    "message": "",
}

def upload_file(file, komoditas):
    if file.filename == '':
        response["message"] = "File Tidak Boleh Kosong"
        return jsonify(response), 400
    if file and allowed_file(file.filename):
        # new_file = file.filename.replace(file.filename, "DataTraining" + komoditas.replace(" ", "") + )

        # if os.path.exists(new_file):
        #     os.remove(new_file)
        
        # Save the file to the specified directory
        file.save(os.path.join(upload_directory, secure_filename(file.filename)))
        convert_data(file, komoditas)
        os.remove("data/" + file.filename)

        data = pd.read_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv")

        if validate_data(komoditas):
            response["data"] = data.to_dict(orient='records')
            response["message"] = "Upload File Sukses"
        else:
            response["data"] = []
            response["message"] = "Data yang diupload tidak sesuai"
    
        return jsonify(response), 200
    else:
        response["message"] = "Format File Tidak Didukung"
        return jsonify(response), 400  

def convert_data(file, komoditas):
  read_file = pd.read_excel("data/"+file.filename)
  data = read_file.to_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv", 
                  index = None,
                  header=True)
  
  return data

def validate_data(komoditas):
    df = pd.read_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv")
    columns_to_check = ['Curah Hujan', 'Harga', 'Produksi']

    all_columns_exist = all(column in df.columns for column in columns_to_check)

    return all_columns_exist