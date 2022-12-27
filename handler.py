# Flask Library -> Conjunto de funções para lidar com requests web
from flask import Flask, request, Response
from rossman.Rossman import Rossman
import pickle
import pandas as pd
import os
import xgboost

model = pickle.load(open('model/model_rossman_1.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

# End Point - envia dados para poder receber
@app.route('/rossman/predict', methods=['POST'])


def rossman_predict():

    test_json = request.get_json()

    # Testando se recebeu algum dado
    if test_json:

        # One Dictionary
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            # Vários jsons concatenados
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
   
        # Instantiate
        pipeline = Rossman()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json') # 200 ->  A requisição deu certo, mas a execução deu errado

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port=port) 