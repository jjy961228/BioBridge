import os
import pandas as pd

class Load_dataset:
    @staticmethod
    def load_dataset():
        # EMR dataset path
        train_path = os.path.join("..", "data_v8_for_access", "train.csv")
        valid_path = os.path.join("..", "data_v8_for_access", "valid.csv")
        test_path = os.path.join("..", "data_v8_for_access", "test.csv")

        train_data = pd.read_csv(train_path, low_memory = False) 
        valid_data = pd.read_csv(valid_path, low_memory = False)
        test_data = pd.read_csv(test_path, low_memory = False)

        train_data = train_data[['ER_DHX','LABEL']] # ER_DHX means Present Illness(PI) of the EMR dataset.
        valid_data = valid_data[['ER_DHX','LABEL']]
        test_data = test_data[['ER_DHX','LABEL']]
        
        train_data = train_data.dropna(axis=0)
        train_data = train_data.reset_index(drop=True)
        valid_data = valid_data.dropna(axis=0)
        valid_data = valid_data.reset_index(drop=True)
        test_data = test_data.dropna(axis=0)
        test_data = test_data.reset_index(drop=True)

        label_frequency = train_data['LABEL'].sum() / len(train_data)

        print('THRESHOLD : {:6f}'.format(label_frequency))
        print('Train missing data : X : {}, Y : {}'.format(train_data.ER_DHX.isna().sum(), train_data.LABEL.isna().sum()))
        print('Valid missing data : X : {}, Y : {}'.format(valid_data.ER_DHX.isna().sum(), valid_data.LABEL.isna().sum()))
        print('Test missing data : X : {}, Y : {}'.format(test_data.ER_DHX.isna().sum(), test_data.LABEL.isna().sum())) 
        
        return train_data, valid_data, test_data, label_frequency
    


