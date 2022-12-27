import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
import sklearn

class Rossman(object):

       def __init__(self):

           
              # self -> class que não pode ser acessada de fora
              self.competition_distance_scaler = pickle.load( open('scalers/comp_distance_scaler.pkl', 'rb'))

              self.competition_time_month_scaler = pickle.load(open('scalers/comp_time_month_scaler.pkl', 'rb'))

              self.year_scaler = pickle.load(open('scalers/year_scaler.pkl', 'rb'))

              self.promo_time_week_scaler = pickle.load(open('scalers/promo_time_week_scaler.pkl', 'rb'))

              self.store_type_scaler = pickle.load(open('scalers/store_type_encoder.pkl', 'rb'))


              state = 1


       def data_cleaning(self, df1):


              ### 2.1 Rename Columns
              cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2','Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

              snakecase = lambda x: inflection.underscore(x)
              cols_new = list(map(snakecase, cols_old))

              # Rename
              df1.columns = cols_new
             
              ### 2.3 Data Types
              df1['date'] = pd.to_datetime(df1['date'])
              
              ### 2.5 Fill NA
              
              # If the competition is too far away we assume there is no c
              df1['competition_distance'] = df1['competition_distance'].apply(lambda x : 200000.0 if math.isnan(x) else x)

              # competition_open_since_month
              df1.loc[df1['competition_open_since_year'].isnull(), 'competition_open_since_year'] = 0
              df1.loc[df1['competition_open_since_month'].isnull(), 'competition_open_since_month'] = 0
              df1.loc[df1['promo2_since_week'].isnull(), 'promo2_since_week'] = 0
              df1.loc[df1['promo2_since_year'].isnull(), 'promo2_since_year'] = 0

              df1['competition_open_since_month'] = df1[['date', 'competition_open_since_month']].apply(lambda x: x['date'].month if x['competition_open_since_month'] == 0 else x['competition_open_since_month'], axis=1)

              # competition_open_since_year 
              df1['competition_open_since_year'] = df1[['date', 'competition_open_since_year']].apply(lambda x: x['date'].year if x['competition_open_since_year'] == 0 else x['competition_open_since_year'], axis=1)
                            
              # promo2_since_week
              df1['promo2_since_week'] = df1[['date', 'promo2_since_week']].apply(lambda x: x['date'].week if x['promo2_since_week'] == 0 else x['promo2_since_week'], axis=1)

              # promo2_since_year
              df1['promo2_since_year'] = df1[['date', 'promo2_since_year']].apply(lambda x: x['date'].year if x['promo2_since_year'] == 0 else x['promo2_since_year'], axis=1)

              # promo_interval              
              month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

              df1['promo_interval'].fillna(0, inplace=True)

              df1['month_map'] = df1['date'].dt.month.map(month_map)

              df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
              
              ### 2.6 Change Type
              df1.dtypes
              df1['competition_open_since_month' ] = df1['competition_open_since_month'].astype(int)
              df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
              df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
              df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

              return df1


       def feature_engineering(self, df2):

              # year
              df2['year'] = df2['date'].dt.year
              # month
              df2['month'] = df2['date'].dt.month
              # day
              df2['day'] = df2['date'].dt.day
              # week of year
              df2['week'] = df2['date'].dt.isocalendar().week
              # year week
              df2['year_week'] = df2['date'].dt.strftime("%Y-%W")

              # weekday
              df2['weekday'] = df2['day_of_week'].apply(lambda x: 1 if x < 6 else 0)

              # competition since
              df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
              df2['competition_since'] = pd.to_datetime(df2['competition_since'])

              # divide by 30 then extract number of months.
              df2['competition_time_month'] = df2.apply(lambda x: int(((x['date'] - x['competition_since'])/30).days), axis=1)


              # promo since
              # Merge the promo year and week (string)
              # datetime.datetime.strptime() - timedelta(days=7)
              df2['promo2_since'] = df2['promo2_since_year'].astype('str') + '-' + df2['promo2_since_week'].astype('str')

              # semana do ano e semana domingo a domingo
              df2['promo2_since'] = df2['promo2_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

              # Subtract from the date and divide by 7 (by week)
              df2['promo2_time_week'] = ((df2['date'] - df2['promo2_since'])/7).apply(lambda x: int(x.days))

              # assortment
              df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

              # state holiday
              df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

              ### 4.1 Filtragem das Linhas

              df2 = df2.loc[(df2['open'] != 0)]

              ### 4.2 Seleção das Colunas
              cols_drop = ['open', 'promo_interval', 'month_map']
              df2 = df2.drop(cols_drop, axis=1)

              return df2

       def data_preparation(self, df5):

              ### 6.1 Rescaling
              ### 6.2 Robust scale_range
              #Diminuir depedência de outliers

              # competition_distance.
              df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)


              # competition_time_month
              df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)
              
              #year
              df5['year'] = self.year_scaler.transform(df5[['year']].values)

              #promo2_time_week'
              df5['promo2_time_week'] = self.promo_time_week_scaler.transform(df5[['promo2_time_week']].values)
              
              ### 6.3 Transformation

              #### 6.3.1 Encoding

              # state_holiday One Hot Encoding
              df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

              # store_type - Label Encoding
              df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])
              
              # assortment - Ordinal Enconding
              assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
              df5['assortment'] = df5['assortment'].map(assortment_dict)

              
              #### 6.3.2 Nature Transformation
              # Trazer a natureza real da sua variável dentro do conjunto de dados.

              # day of week
              df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi/7)))
              df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi/7)))

              # month
              df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi/12)))
              df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi/12)))

              # day
              df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi/30)))
              df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi/30)))

              # week of year
              df5['week_sin'] = df5['week'].apply(lambda x: np.sin(x * (2 * np.pi/52)))
              df5['week_cos'] = df5['week'].apply(lambda x: np.cos(x * (2 * np.pi/52)))

              cols_selected = ['store',
                               'promo',
                               'store_type',
                               'assortment',
                               'competition_distance',
                               'competition_open_since_month',
                               'competition_open_since_year',
                               'promo2',
                               'promo2_since_week',
                               'promo2_since_year',
                               'weekday',
                               'competition_time_month',
                               'promo2_time_week',
                               'day_of_week_sin',
                               'day_of_week_cos',
                               'month_cos',
                               'month_sin',
                               'day_sin',
                               'day_cos',
                               'week_cos']

              return df5[cols_selected]
       
       def get_prediction(self, model, original_data, test_data):

              # prediction
              pred = model.predict(test_data)

              # join pred into the original data
              original_data['prediction'] = np.expm1(pred)

              return original_data.to_json(orient='records', date_format='iso')
