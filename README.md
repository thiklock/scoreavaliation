# Introdução
Levantamento de histórico
# Objetivo
Declaração do Objetivo
# Ferramentas Aplicadas e Sujestões
Validar em Dev, realizar o deploy em prod
# Planning
Time Management:
    There is 72 hours to achive the goal. Me, the Executor,  has only 24 hours i the day
    which 8 are destined to sleeping, 2 for general activities and 9 to work. This means I have only 5 hours a day to apply to the project. 

    The Deadline for the conclusio is: 19:00 hours of may, 7 th of 2021.
    Human Resources: 5 hours per day, this is: 15:00 hours of me.
    Tools: Personal Notebok and open source libraries.

    Within only 15 hours of project and in a 3 days time basis the project shall be conducted within the Minimum Vaiable Product model, doing the essentially necessary and wraping everything in the most dry way, so the final client (reviewer) shall run and receive what was required. Somethings will fall short to a deployable infrastructute but nevertheless it should proove that the proposed model generate good results. 

    First 5 hours: Planning, Data Exploration And Data Cleaning.
        Should explore, view and clean the current.
    Second 5 hours: Feature Engineering and Model Aplication
        Should create new features to better serve Machine Learning Models, Apply at least 3 models to the project. 
    Last 5 Hours: Model Avaliation and Presentation:
        At the final sprint of the project the focus shall be on the evalutation of the model and on the presentation. As this project shall be reviewd by someone who understands python the main effort will be applied to leep it dry and comment anything necessary but a presentation should be added if there is time for it.

# Desenvolvimento

Step by Step

Pre Work:
    Create virtual enviroment with python 3.6
    Create debugger file module at Visual code Studio
    Create git repository

Data Exploration:
    Action:
    The data was read, described and the null values were counted.
    Result:
    There are 10 columns and 110 000 rows of data.
    The types of the columns are described and in resume:

            #   Column                                 Non-Null Count   Dtype  
        ---  ------                                 --------------   -----  
         0   inadimplente                           110000 non-null  int64  
         1   util_linhas_inseguras                  110000 non-null  float64
         2   idade                                  110000 non-null  int64  
         3   vezes_passou_de_30_59_dias             110000 non-null  int64  
         4   razao_debito                           110000 non-null  float64
         5   salario_mensal                         88237 non-null   float64
         6   numero_linhas_crdto_aberto             110000 non-null  int64  
         7   numero_vezes_passou_90_dias            110000 non-null  int64  
         8   numero_emprestimos_imobiliarios        110000 non-null  int64  
         9   numero_de_vezes_que_passou_60_89_dias  110000 non-null  int64  
         10  numero_de_dependentes                  107122 non-null  float64
        dtypes: float64(4), int64(7)
        memory usage: 9.2 MB

    Every column looks great with efew asonable non null    values but two:

        There are 21760 null values in the column   "salario_mensal".
        There are 2878 null values in the column    "numero_de_dependentes".

    There are many ways for daeling with those missing values, to drop the columns or the row with missing values could be a solution for the "numero_de_dependentes" column as only as small part suffers (2,6%), but for the other column "numero_de_dependentes" the impact would be of 19,79% of the total rows.

    The project can procede with the first drop but not with the second.

    With this being said the best solution for this should be not to drop the rows but reither fill with averages (low impact or either to drop the column it self.) As the time table it is very short, the best option it is to drop it now, procede with the task and return to it if there is enough time.

    First thing in a analysis like this, should undersntand how balanced is our dataset, i.e. how is the data balanced between both of the possibilities: overdue(1) or not(0). 

    The dataset is heavely inbalanced, showing 7331 overdued samples for 102669 not overdued people.

    As the age feature showed a grater impact we should invetigate further the impact of it.

    With the funciton age_explorer thi distribution of the age is ploted showing a heavy concentration around the age of 40 years old. The minimum age f 0 and the maximum of 109 it is also detected and printed by the function.

        In this same function ages of 0 and some other incosistencies are droped as a way to improve data quality.

        A new line chart of the distribution it is ploted now showing the density comparing the diference between valuew of inadimplentes and not inadimplentes.

        To investigater further the relation between age and inadimplente the clients were grouped by age groups and the mean average of inadimplente and mean age was shown by age group.

        It is very important to watch thi graph carefully. There is a clear trend ins this dataset: There are more clients who fail in the two younger groups than in the others.

        LAST TWO FUNCTIONS

        There are a infinite number of thing taht should be done with the purpose of a better analsys of the dataset as a compation, feature by feature, compairng how the column inadimplente behaves feature by feature.  
        



    



    


    
# Instruções

# Conclusão


### Sources
https://hal.archives-ouvertes.fr/hal-02507499v2/document

https://medium.com/analytics-vidhya/what-is-balance-and-imbalance-dataset-89e8d7f46bc5

