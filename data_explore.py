import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    print('Droping Columns:')
    print(cols_to_drop)
    df_with_droped_cols = df.drop(columns=cols_to_drop)
    print('Columns Droped.')
    return df_with_droped_cols

df_droped_cols = df_drop_col(df_raw)

print(df_droped_cols)
def pairplot(df):
    ploted_matplot_lib = pd.plotting.scatter_matrix(df, figsize=(10,10), marker = 'o', hist_kwds = {'bins': 10}, s = 60, alpha = 0.8)
    plt.savefig('raw_data_with_droped_cols_matplotlib.png')

    ploted_seaborn = sns.pairplot(df,hue='inadimplente')
    plt.savefig("seaborn_plot.png")
    plt.show()

# droped_cols_plot = pairplot(df_droped_cols)

# /home/tik/chal/dataminer/source/treino.csv

def balance_evaluation(df):
    print('counting inadimplentes of util_linhas_inseguras')
    count_inadimplente = df[df['inadimplente']==1]['inadimplente'].count()
    print(count_inadimplente)
    print('counting inadimplentes of numero_emprestimos_imobiliarios')
    count_not_inadimplente = df[df['inadimplente']==0]['inadimplente'].count()
    print(count_not_inadimplente)


amount_ina = balance_evaluation(df_droped_cols)

# Data Explore ??Evaluate duplicates??

# Coorelation

def correlate(df):
    print('Evaluating Correlations')
    correlations = df.corr()['inadimplente'].sort_values()
    print('The Most Positive Correlations:\n', correlations.tail(3))
    print('\nThe Most Negative Correlations:\n', correlations.head(3))
    correlate = df.corr()
    # Heatmap of correlations
    sns_plot = sns.heatmap(correlate, cmap = plt.cm.seismic, vmin = -0.30, annot = True, vmax = 0.70)
    # sns_plot.savefig("output.png")
    sns_plot.figure.savefig('correlations.png')



def age_explorer(df):
    plt.style.use('fivethirtyeight')
    # Plot the distribution of ages in years
    plt.hist(df['idade'], edgecolor = 'k')
    plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
    print('min age {} max age {}'.format(df['idade'].min(), df['idade'].max()))
    print('age <20 {}, age >99 {}'.format(len(df[df['idade']>99]),len(df[df['idade']<20])))
    correlation = correlate(df_droped_cols)
    



age_explored = age_explorer(df_droped_cols)