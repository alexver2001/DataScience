import flask
import numpy as np
from flask import render_template, send_file, send_from_directory
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import  StandardScaler
import pandas as pd
import os
app = flask.Flask(__name__, template_folder = 'templates')
data_path = r'./Rezult_data/'
data_file = 'Rezult_data.xlsx'
if os.path.exists('./Rezult_data/Rezult_data.xlsx'):
    Rezult = pd.read_excel('./Rezult_data/Rezult_data.xlsx',index_col=0)
else:
    Rezult = pd.DataFrame(columns=['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'Модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп, %_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2',  'Потребление смолы, г/м2',
       'Угол нашивки,град', 'Шаг нашивки', 'Плотность нашивки', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа'])

Name_InParam = ['Param_1','Param_2', 'Param_3', 'Param_4', 'Param_5', 'Param_6','Param_7','Param_8', 'Param_9', 'Param_10', 'Param_11']
# загрузка модели
loaded_model = keras.models.load_model(r'./ML_Model/')
with open(r'./ML_Model/scaler_std.pkl', 'rb') as f:
    scaler_std = pickle.load(f)
with open(r'./ML_Model/scaler_std_y.pkl', 'rb') as f:
    scaler_std_y = pickle.load(f)
data_path = r'./Rezult_data/'
data_file = 'Rezult_data.xlsx'

@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('index.html', Columns = Rezult.columns, Name_InParam = Name_InParam, Columns_input = Rezult.columns.tolist()[0:11], Columns_output =Rezult.columns.tolist()[11:13], Predict_Value = ['-', '-'], Data_Predict = Rezult)


@app.route('/Predict', methods = ['POST', 'GET'])
def predict():
    if flask.request.method == 'POST':
        # считывание из формы параметров
        X_pred = []
        for param in Name_InParam:
            X_pred.append(float(flask.request.form[param]))
        X_pred_std = scaler_std.transform(np.array(X_pred).reshape(-1, 11))
        y_pred_std = loaded_model.predict(X_pred_std)
        y_pred = scaler_std_y.inverse_transform(y_pred_std)
        y_pred_1 = y_pred[0,0]
        y_pred_2 = y_pred[0,1]
        exp= X_pred.copy()
        exp.append(y_pred_1)
        exp.append(y_pred_2)
        Rezult.loc[len(Rezult.index)] = exp
        Rezult.to_excel(data_path + data_file)
        return render_template('index.html', Columns = Rezult.columns, Name_InParam = Name_InParam, Columns_input = Rezult.columns.tolist()[0:11], Predict_Value = [y_pred_1,y_pred_2], Columns_output =Rezult.columns.tolist()[11:13], Data_Predict = Rezult )

    if flask.request.method == 'GET':

        return render_template('index.html', Columns = Rezult.columns, Name_InParam = Name_InParam, Columns_input = Rezult.columns.tolist()[1:12], Predict_Value = [0,0], Data_Predict = Rezult)

@app.route('/about', methods = ['POST', 'GET'])
def about():
    if flask.request.method == 'GET':
        return render_template('about.html')

@app.route('/save_results')
def download():
    if flask.request.method == 'GET':
       response = send_from_directory('./Rezult_data','Rezult_data.xlsx', mimetype='application/vnd.ms-excel')
       response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
       return response

@app.route('/clear_results')
def Clear():
    if flask.request.method == 'GET':
       Rezult = pd.DataFrame(columns=['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'Модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп, %_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2',  'Потребление смолы, г/м2',
       'Угол нашивки,град', 'Шаг нашивки', 'Плотность нашивки', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа'])
       Rezult.to_excel(data_path + data_file)
       return render_template('index.html', Columns = Rezult.columns, Name_InParam = Name_InParam, Columns_input = Rezult.columns.tolist()[1:12], Predict_Value = [0,0], Data_Predict = Rezult)
if __name__ == '__main__':
    app.run()