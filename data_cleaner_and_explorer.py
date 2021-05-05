import numpy as np

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
    # # Heatmap of correlations
    # sns_plot = sns.heatmap(correlate, cmap = plt.cm.seismic, vmin = -0.30, annot = True, vmax = 0.70)
    # # sns_plot.savefig("output.png")
    # sns_plot.figure.savefig('correlations_plot.png')

correlation = correlate(df_droped_cols)

def age_explorer(df):
    # plt.style.use('fivethirtyeight')
    # Plot the distribution of ages in years
    fig = plt.hist(df['idade'], edgecolor = 'k')
    plt.show()
    plt.savefig("abc.png")
    # age_plot.plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count')
    # age_plot.plt.savefig('age_dist.png')
    print('minimun age {} maximum age {}'.format(df['idade'].min(), df['idade'].max()))
    print('Clients with age < 20 {}, clients with age > 99 {}'.format(len(df[df['idade']>99]),len(df[df['idade']<20])))

    plt.figure(figsize = (10, 8))
    # KDE plot of loans that were repaid on time
    sns.kdeplot(df.loc[df['inadimplente'] == 0, 'idade'], label = 'target == 0')
    # KDE plot of loans which were not repaid on time
    sns.kdeplot(df.loc[df['inadimplente'] == 1, 'idade'], label = 'target == 1')
    # Labeling of plot
    plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
    plt.savefig("distribution_of_ages.png")
    # Age information into a separate dataframe
    age_data = df[['inadimplente', 'idade']]
    # Bin the age data
    age_data['age_binned'] = pd.cut(age_data['idade'], bins = np.linspace(20, 60, num = 6))
    print(age_data.head(10))
    age_groups  = age_data.groupby('age_binned').mean()
    print(age_groups)
    plt.figure(figsize = (8, 8))
    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.index.astype(str), 100 * age_groups['inadimplente'])
    # Plot labeling
    plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group')
    plt.savefig('Failure_to_Repay_by_Age_Group.png')

age_explored = age_explorer(df_droped_cols)

def open_cred_lines_explorer(df):
    plt.figure(figsize = (10, 8))
    sns.kdeplot(df.loc[df['inadimplente'] == 0, 'numero_linhas_crdto_aberto'], label = 'target == 0')
    sns.kdeplot(df.loc[df['inadimplente'] == 1, 'numero_linhas_crdto_aberto'], label = 'target == 1')
    plt.xlabel('number of open credit lines'); plt.ylabel('Density'); plt.title('Distribution of number of open credit lines')
    plt.savefig('open_cred_lines_dist.png')


open_lined_explored = open_cred_lines_explorer(df_droped_cols)

def real_state_loans_explorer(df):
    print('creating number real estate loans plot.')
    plt.figure(figsize = (10, 8))
    sns.kdeplot(df.loc[df['inadimplente'] == 0, 'numero_emprestimos_imobiliarios'], label = 'target == 0')
    sns.kdeplot(df.loc[df['inadimplente'] == 1, 'numero_emprestimos_imobiliarios'], label = 'target == 1')
    plt.xlabel('number real estate loans'); plt.ylabel('Density'); plt.title('Distribution of number real estate loans');
    plt.savefig('number_real_state_loans.png')
    print('number real estate loans plot created.')

number_real_state_loans_explored = real_state_loans_explorer(df_droped_cols)

def generate_cleaned_data(df):
    cleaned_date = df
    df.to_csv(r'cleaned_test.csv')

cleaned_data = generate_cleaned_data(df_droped_cols)


