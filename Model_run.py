from Traffic_Citations import Traffic_Citations_data_cleaning
from s3_data_ingest import downloadData


class Model_run:

    def __init__(self):
        self.Tc=Traffic_Citations_data_cleaning()
        self.s3=downloadData()
        self.subset=['Issue time','RP State Plate','Body Style','Color','Location','Route','Agency','Violation Description','Fine amount','Latitude','Longitude']
        self.columns= ['Meter Id','Marked Time','VIN']

    def Run_Model(self):
        print("######################################DOWNLOADING DATA FROM s3####################################################")
        self.s3.s3_download_data()
        print("######################################LOADING INITIAL DATAFRAME####################################################")
        Dataframe=self.Tc.read_data()
        print("######################################DATAFRAME LOADED####################################################")
        print("######################################PERFORMING MAGIC INSIDE####################################################")
        Dataframe2=self.Tc.dataframe_nullValues_removal_logic(Dataframe,self.subset,self.columns)
        df_train_test,df_predict=self.Tc.objectEncoder(Dataframe2)
        X_train,X_test,y_train,y_test,X_predict,y_predict,X_train_test,y_train_test=self.Tc.split_train_test(df_train_test,df_predict)
        print("######################################TRAINING MODEL AND PRINTING CV SCORES####################################################")
        self.Tc.train_model(X_train,y_train,X_train_test,y_train_test)

    # def Run_data_cleaning(self,Dataframe):
    #     Dataframe2 = self.Tc.dataframe_nullValues_removal_logic(Dataframe, self.subset, self.columns)
    #     print(Dataframe2)
    #     df_train_test, df_predict = self.Tc.objectEncoder(Dataframe2)
    #     X_predict = df_predict.drop(columns=['Issue Date', 'Make', 'Common make'])
    #     return X_predict

if __name__=="__main__":
    a=Model_run()
    a.Run_Model()