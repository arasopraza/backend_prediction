import pandas as pd
import numpy as np
from flask import jsonify
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

response = {}

def train_data(komoditas):
  df = pd.DataFrame(pd.read_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv"))
  
  if detect_outlier(df, "Curah Hujan") is None or detect_outlier(df, "Harga") is None or detect_outlier(df, "Produksi") is None :
    detect_null_or_empty(df)
    # Train 
    modeling(df)
    y_pred = modeling(df)[4]

    response["korelasi_hujan_kepada_harga"] = {
      "nama": modeling(df)[1],
      "nilai": modeling(df)[0]
    }
    response["korelasi_produksi_kepada_harga"] = {
      "nama": modeling(df)[3],
      "nilai": modeling(df)[2]
    }
    response["hasil_prediksi"] = y_pred.tolist()
  
  else:
    detect_null_or_empty(df)
    columns_to_check = ['Curah Hujan', 'Harga', 'Produksi']
    for column in columns_to_check:
      binning_data(df, column)

    modeling(df)
    y_pred = modeling(df)[4]
    response["korelasi_hujan_kepada_harga"] = {
      "nama": modeling(df)[1],
      "nilai": modeling(df)[0]
    }
    response["korelasi_produksi_kepada_harga"] = {
      "nama": modeling(df)[3],
      "nilai": modeling(df)[2]
    }
    response["hasil_prediksi"] = y_pred.tolist()
    
    return jsonify(response)

  return jsonify(response)
  
def detect_null_or_empty(df):
  data = []
  null_rows = df[df.isnull().any(axis=1)]
  for idx, row in null_rows.iterrows():
    row_dict = {}
    for column in df.columns:
      row_dict[column] = row[column] if pd.notna(row[column]) else 'nan'
    data.append(row_dict)

  if not null_rows.empty:
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return data
  
def detect_outlier(df, column):
  data = df[column].sort_values().reset_index(drop= True)
  
  # Compute the quartiles
  q1 = (data.loc[2] + data.loc[3]) / 2
  q3 = (data.loc[8] + data.loc[9]) / 2
  IQR = q3 - q1
  lower_bound = q1 - (1.5*IQR)
  upper_bound = q3 + (1.5*IQR)

  outlier = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
  if not outlier.empty:
    return outlier
  else:
    return None

def binning_data(df, column):
  data = df[column].sort_values().reset_index(drop= True)
  data.to_numpy()

  split_data = np.array_split(data, 3)
  bin1 = split_data[0]
  bin2 = split_data[1]
  bin3 = split_data[2]

  mean_bin1 = round(bin1.mean())
  mean_bin2 = round(bin2.mean())
  mean_bin3 = round(bin3.mean())

  df[column] = df[column].mask(df[column] <= bin1[3], mean_bin1)
  df[column] = df[column].mask((df[column] > bin1[3]) & (df[column] < bin3[8]), mean_bin2)
  df[column] = df[column].mask(df[column] >= bin3[8], mean_bin3)

  return df

def normalization_data(df):
  df["Curah Hujan"] = (df["Curah Hujan"] - df["Curah Hujan"].min()) / (df["Curah Hujan"].max() - df["Curah Hujan"].min())
  df["Produksi"] = (df["Produksi"] - df["Produksi"].min()) / (df["Produksi"].max() - df["Produksi"].min())

  curah_hujan_produksi = df["Curah Hujan"] * df["Produksi"]
  df["Curah Hujan dan Produksi"] = curah_hujan_produksi


def modeling(df):
  keterangan_korelasi_hujan = ""
  keterangan_korelasi_produksi = ""

  normalization_data(df)
  x = df[["Produksi", "Curah Hujan dan Produksi"]]
  y = df["Harga"]

  regressor = LinearRegression()
  regressor.fit(x, y)
  
  # print('Intercept: \n', regressor.intercept_)
  # print('Coefficients: \n', regressor.coef_)

  y_pred = regressor.predict(x)
  korelasi_hujan, _ = pearsonr(df["Curah Hujan"], df["Harga"])
  korelasi_produksi, _ = pearsonr(df["Produksi"], df["Harga"])

  if 0 <= korelasi_hujan < 0.2:
    keterangan_korelasi_hujan = "Sangat Lemah"
  elif 0.2 <= korelasi_hujan < 0.4:
    keterangan_korelasi_hujan = "Lemah"
  elif 0.4 <= korelasi_hujan < 0.6:
    keterangan_korelasi_hujan = "Sedang"
  elif 0.6 <= korelasi_hujan < 0.8:
    keterangan_korelasi_hujan = "Kuat"
  elif 0.8 <= korelasi_hujan <= 1.0:
    keterangan_korelasi_hujan = "Sangat Kuat"
  else:
    print("Value is out of the expected range")

  if 0 <= korelasi_produksi < 0.2:
    keterangan_korelasi_produksi = "Sangat Lemah"
  elif 0.2 <= korelasi_produksi < 0.4:
    keterangan_korelasi_produksi = "Lemah"
  elif 0.4 <= korelasi_produksi < 0.6:
    keterangan_korelasi_produksi = "Sedang"
  elif 0.6 <= korelasi_produksi < 0.8:
    keterangan_korelasi_produksi = "Kuat"
  elif 0.8 <= korelasi_produksi <= 1.0:
    keterangan_korelasi_produksi = "Sangat Kuat"
  else:
    print("Value is out of the expected range")
  
  return (korelasi_hujan, keterangan_korelasi_hujan, korelasi_produksi, keterangan_korelasi_produksi, y_pred)