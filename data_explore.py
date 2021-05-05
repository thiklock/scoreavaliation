import pandas as pd

from library import Data_Cleaner

raw_data_path = '/home/tik/chal/dataminer/source/treino.csv'

def data_loader(path):
    print('loading file from' + path)
    df = pd.read_csv(path)
    print(df.head(5))
    return df

df_raw = data_loader(raw_data_path) # Loading the raw file into a df variable

def data_describe(df_raw):
    '''
    This function prints a series of information
    from the data received.
    '''
    print('describing raw dara')
    df_described = df_raw.describe()
    print(df_described)

    print('Data types:')
    data_types = df_raw.dtypes
    print(data_types)

    print('data info')
    data_info = df_raw.info()
    print(data_info
    )
    print('Count of nulls:')
    count_of_nulls = df_raw.isnull().sum()
    print(count_of_nulls)
    return count_of_nulls

df_described = data_describe(df_raw)



def df_drop_col(df):
    cols_to_drop = ['salario_mensal', 'salario_mensal']
    
df_droped_cols = df_drop_col(df_described)
# /home/tik/chal/dataminer/source/treino.csv