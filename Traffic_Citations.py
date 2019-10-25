import os
import sys
import pandas as pd
import numpy as np
from config.Json_read import JsonObj
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class Traffic_Citations_data_cleaning:

    def __init__(self):
        obj = JsonObj()
        self.file_name = obj.get_config_obj('file_name')
        self.data_store_location = obj.get_config_obj('data_store_location')
        self.le = LabelEncoder()

    def read_data(self):
        df_parking_citations = pd.read_csv(self.data_store_location+self.file_name,header=True,encoding='utf-8')
        return df_parking_citations

    def dataframe_nullValues_removal_logic(self,dataframe,subset,columns):
        df_parking_citations=dataframe.dropna(axis = 'index', subset = subset)
        df_parking_citations = df_parking_citations.drop(columns = columns)
        df_parking_citations['Plate Expiry Date'].fillna(value= 202012.00, inplace= True )
        df_parking_citations['Ticket number'] = df_parking_citations['Ticket number'].astype(str).str.replace('D', '')
        df_parking_citations['Issue Date'] =  pd.to_datetime(df_parking_citations['Issue Date'], format='%Y-%m-%dT%H:%M:%S')
        df_parking_citations['Time of Day'] =  [ 'AM' if x < 1200 else 'PM' for x in df_parking_citations['Issue time']]
        return df_parking_citations

    def objectEncoder(self,dataframe):
        categorical_feature_mask = dataframe.dtypes == object
        categorical_cols = dataframe.columns[categorical_feature_mask].tolist()
        categorical_cols.remove('Make')
        df_encode = dataframe
        df_encode[categorical_cols] = df_encode[categorical_cols].apply(lambda col: self.le.fit_transform(col))
        x = dataframe['Fine amount']
        df_encode['Fine amount'] = (x - x.min()) / (x.max() - x.min())
        df_train_test = df_encode.dropna(axis='index', subset=['Make'])
        df_predict = df_encode[df_encode['Make'].isnull()]
        df_common_make = df_train_test[['Ticket number', 'Make']].groupby(['Make']) \
            .agg('count') \
            .sort_values('Ticket number', ascending=False) \
            .head(25) \
            .reset_index()
        l_common_make = df_common_make['Make'].tolist()
        df_train_test['Common make'] = [1 if x in l_common_make else 0 for x in df_train_test['Make']]
        df_predict['Common make'] = np.nan
        return df_train_test,df_predict

    def split_train_test(self,df_train_test,df_predict):
        X_train_test = df_train_test.drop(['Issue Date', 'Make', 'Common make'], axis=1)
        y_train_test = df_train_test['Common make']
        X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, random_state=200, test_size=0.2)
        X_predict = df_predict.drop(['Issue Date', 'Make', 'Common make'], axis=1)
        y_predict = df_predict['Common make']
        return X_train, X_test, y_train, y_test,X_predict,y_predict,X_train_test,y_train_test

    def train_model(self,X_train,y_train,X_train_test,y_train_test):
        knn = KNeighborsClassifier(n_neighbors=100)
        knn.fit(X_train, y_train)
        pickle.dump(knn,open('model/model.pkl','wb'))
        kfold = KFold(n_splits=5, random_state=None)
        cv_scores = cross_val_score(knn, X_train_test, y_train_test, cv=kfold)
        return print(cv_scores)