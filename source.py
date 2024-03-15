import sqlite3
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import  SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.metrics import f1_score




from collections import Counter
import itertools




def create_db(path=r"C:\Users\USUARIO\OneDrive - Universidad de Antioquia\Analitica_3"):
    """
    Funcion para la creacion del Database

    Parametros
    ------------
        path: direccion local de los archivos como una raw string

    """
    
    
        
    if os.path.exists("data/human_db"):
        pass
    
    else:
        
        
        
        os.mkdir("data")
        file_names = [i for i in os.listdir(path) if os.path.splitext(i)[1] == ".csv"]
        conn = sqlite3.connect("data/human_db")
    
        for x in file_names:
            pd.read_csv(os.path.join(path, x)).to_sql(os.path.splitext(x)[0], conn, if_exists="replace")
    
        cursor = conn.cursor()
    
        ### Completar valores nulos 
        cursor.execute("UPDATE employee_survey_data SET EnvironmentSatisfaction = (SELECT ROUND(AVG(EnvironmentSatisfaction), 0) FROM employee_survey_data) WHERE EnvironmentSatisfaction IS NULL")
        cursor.execute("UPDATE employee_survey_data SET JobSatisfaction = (SELECT ROUND(AVG(JobSatisfaction), 0) FROM employee_survey_data) WHERE JobSatisfaction IS NULL")
        cursor.execute("UPDATE employee_survey_data SET WorkLifeBalance = (SELECT ROUND(AVG(WorkLifeBalance), 0) FROM employee_survey_data) WHERE WorkLifeBalance IS NULL")
    
        ## Crear nueva tabla
        cursor.execute("CREATE TABLE IF NOT EXISTS employee_survey_data_up AS SELECT EmployeeID, DateSurvey AS Date, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance FROM employee_survey_data")
        ## Crear nuevo index
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_employee_survey_data_up_EmployeeID ON employee_survey_data_up(EmployeeID)")
    
        ## Eliminar valores nulos
        cursor.execute("DELETE FROM general_data WHERE NumCompaniesWorked IS NULL")
        cursor.execute("DELETE FROM general_data WHERE TotalWorkingYears IS NULL")
        
        # Eliminar despedidos
        # cursor.execute("DELETE FROM retirement_info WHERE retirementType = 'Fired'")
        
        ## Crear nueva tabla elimiando Over18, EmployeeCount, StandardHours
        cursor.execute("CREATE TABLE IF NOT EXISTS general_data_up AS SELECT EmployeeID, InfoDate AS Date, Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, Gender, JobLevel, JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager	FROM general_data")
        ## Crear nuevo index
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_general_data_up_EmployeeID ON general_data_up(EmployeeID)")
    
        ## Crear nueva tabla
        cursor.execute("CREATE TABLE IF NOT EXISTS manager_survey_up AS SELECT EmployeeID, SurveyDate AS Date, JobInvolvement, PerformanceRating FROM manager_survey")
        ## Crear nuevo index
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_manager_survey_up_EmployeeID ON manager_survey_up(EmployeeID)")
    
        ## Completar valores nulos
        cursor.execute("UPDATE retirement_info SET resignationReason = 'Fired' WHERE resignationReason IS NULL;")
    
        ## Crear nueva tabla
        cursor.execute("CREATE TABLE IF NOT EXISTS retirement_info_up AS SELECT EmployeeID, Attrition, retirementDate AS Date, retirementType, resignationReason FROM retirement_info")
        ## Crear nuevo index
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_retirement_info_up_EmployeeID ON retirement_info_up(EmployeeID)")
    
  
        ## Cambio de catiegorias con muy poca influencia en las variables
        cursor.execute("UPDATE manager_survey_up SET JobInvolvement = 0 WHERE JobInvolvement IN (1, 2)")
        cursor.execute("UPDATE manager_survey_up SET JobInvolvement = 1 WHERE JobInvolvement IN (3, 4)")
        cursor.execute("UPDATE general_data_up SET Education = 4 WHERE Education = 5")
        
            
        ## Crear tabla con toda la informacion
        cursor.execute("CREATE TABLE IF NOT EXISTS df AS SELECT EmployeeID, Date, CAST(EnvironmentSatisfaction AS TEXT) AS EnvironmentSatisfaction, CAST(JobSatisfaction AS TEXT) AS JobSatisfaction, CAST(WorkLifeBalance AS TEXT) AS WorkLifeBalance, Age, BusinessTravel, Department, DistanceFromHome, CAST(Education AS TEXT) Education, EducationField, Gender, CAST(JobLevel AS TEXT) JobLevel, CAST(JobRole AS TEXT) JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, CAST(StockOptionLevel AS TEXT) AS StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, CAST(JobInvolvement AS TEXT) AS JobInvolvement, CAST(PerformanceRating AS TEXT) AS PerformanceRating, retirementType, resignationReason, Attrition FROM (SELECT * FROM general_data_up t1 INNER JOIN  employee_survey_data_up t2 ON t1.EmployeeID = t2.EmployeeID AND strftime('%Y', t1.Date) = strftime('%Y', t2.Date) INNER JOIN manager_survey_up t3 ON t2.EmployeeID = t3.EmployeeID AND strftime('%Y', t2.Date) = strftime('%Y', t3.Date) LEFT JOIN retirement_info_up t4 ON t3.EmployeeID = t4.EmployeeID AND strftime('%Y', t3.Date) = strftime('%Y', t4.Date))")
        ## Llenar valores nulos de la nueva tabla
        cursor.execute("UPDATE df SET retirementType = 'Active', resignationReason = 'Not applicable' WHERE retirementType IS NULL")
        cursor.execute("UPDATE df SET Attrition = '0' WHERE Attrition IS NULL")
        cursor.execute("UPDATE df SET Attrition = '1' WHERE Attrition = 'Yes'")
        
        cursor.execute("DELETE FROM df WHERE EmployeeID IN (SELECT t1.EmployeeID FROM (SELECT * FROM df WHERE resignationReason <> 'Not applicable' AND strftime('%Y', Date) = '2015') t1 INNER JOIN (SELECT * FROM df WHERE strftime('%Y', Date) = '2016') t2 ON t1.EmployeeID = t2.EmployeeID) AND strftime('%Y', Date) = '2016';")

        
        
        conn.commit()
        
        
        
            
        conn.close()
    


def create_df(query, index=False):
    """
    Funcion para la creacion del Data frame desde la base de datos, recibe una consulta 
    SQL y devulve un Data Frame de pandas

    Parametros
    ------------
        query:  Consulta en SQL de la base de datos requerida
        index: Use EmployeedID as index. Default False

    """

    conn = sqlite3.connect("data/human_db")

    if index:
        df = pd.read_sql(query, conn, index_col="EmployeeID", parse_dates=["Date"])
    else:
        df = pd.read_sql(query, conn)
        
    conn.close()
    return df


def heatmap(df=..., annot=False):
    """
    Funcion para el mapa de correlacion de las variables numericas de un Data Frame
    devuele una figura de matplotlib
    
    Parametros
    ------------
        df: DataFrame con la informacion a dibujar
        annot: Mostrar valores en el grafico
    
    """
    
    
    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"
    
    corr = df._get_numeric_data().corr().round(2)
    
    
    sns.heatmap(corr.abs(), annot=annot);
        
        
        
def pieplot(df=..., groupby=..., count_col="Attrition"):
    """
    Funcion para la visualizacion de la proporcion de empleados por clase a predecir
    Devuelve una figura de Matplotlib
    
    Parametros
    -------------
        df: DataFrame con la informacion
        groupby: columna por la cual agrupar
        count_col: Variable objetivo. Default Attrition
    """
    
    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"
    
    groupby = ["Attrition", df["Date"].dt.strftime("%Y")] if groupby == ... else groupby

    table = df.groupby(groupby, as_index=False)[count_col].value_counts().drop(columns=count_col)
    
    
    plt.figure(figsize=(12, 12))

    plt.subplot(1,3,1)
    plt.pie(x=table.loc[table["Date"] == "2016", "count"], autopct='%1.1f%%', 
                    labels=["Stayed", "Left"], 
                    explode=(0.13, 0), shadow=True, labeldistance=.5, pctdistance=1.3);
    plt.title("Proporcion de empleados despedidos \no que renunciaron 2016", loc="left")
    
    plt.subplot(1,3,3)
    plt.pie(x=table.loc[table["Date"] == "2015", "count"], autopct='%1.1f%%', 
                    labels=["Stayed", "Left"], 
                    explode=(0.6, 0), shadow=True, labeldistance=.5, pctdistance=1.3);
    plt.title("Proporcion de empleados despedidos \no que renunciaron 2015", loc="right")
    

    
    
def histograms(df=..., nrows=4, ncols=3, figsize=(12, 12), var_obj="Attrition"):
    """
    Funcion para la visualizacion de los histogramas de las variables numericas
    
    Parametros
    ------------
        df: Dataframe con la informacion
        nrows : Numero de filas del subplot. Default
        ncols: Numero de columnas del subplot. Default 3
        figsize: Tamano de la nueva figura. Deafult (12, 12)
    """
    
    
    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"

    table = pd.concat([df._get_numeric_data(), df[var_obj]], axis=1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    x = 0
    y = 0

    for c in range(len(table.columns)-1):

        if table.columns.values[c] == var_obj:
            break
        
        sns.histplot(data=table, x=table.columns.values[c], hue=var_obj, ax=axes[x, y], kde=True, bins=10)
        
        if y!=0 and y%2 == 0:
            y = 0
            x += 1
        else:
            y += 1
    
    plt.tight_layout()



def scatterplots(df=..., feats=["YearsAtCompany", "YearsWithCurrManager", "YearsSinceLastPromotion", "TotalWorkingYears", "Age"], nrows=4, ncols=3):
    
    """
    Funcion para visualizacion de variables altamente correlacionadas
    
    Parametros:
    --------------
        df: Data frame con los datos a visualizar
        feats: Columnas altamente correlacionadas
        nrows: Filas la new grafica
        ncols = Columnas de la nueva grafica
    """

    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    i = 0
    j = 0
    
    for x, y in itertools.combinations(feats, r=2):
    
        df[[x, y]].plot.scatter(x=x, y=y, ax=axes[i, j], xlabel=f'{x}')
    
        if j!=0 and j%2 == 0:
            j = 0
            i += 1
        else:
            j += 1
    
    fig.suptitle("Mapas de dispersion para variables de alta correlacion")
    plt.tight_layout()



def barcharts(df=..., normalize=True, obj_col="Attrition", rotation=90, only=False, targ_col=...):
    """
    Funcion para la visualizacion de los histogramas de variables categoricas
    
    Parametros:
    --------------
        df: Data frame con la informacion
        normalize: Usar o no el promedio. Default True
        obj_col: Variable objetivo
        rotacion: rotacion de los ticks de la grafica
        only: True si solo se quiere visualiizar una variable. Default False
        targ_col: Columna a visualizar, solo sirve si only es True
    
    """
    
    
  
    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"  



    data = df.select_dtypes(include=["object"])


    if only:
  
      assert targ_col in data.columns, "Columna no encontrada"
  
      if normalize:
  
        table = data.groupby(obj_col, as_index=False)[targ_col].value_counts(normalize=normalize).round(2)
        order = sorted(table.iloc[:,1].unique())
        ax = sns.barplot(data=table, x=table.columns[1], y="proportion", hue=obj_col, order=order)
  
        if len(order[0]) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
        
        for i in ax.containers:
          ax.bar_label(i,)
  
      else:
  
        table = data.groupby(obj_col, as_index=False)[targ_col].value_counts(normalize=normalize).round(2)
        order = sorted(table.iloc[:,1].unique())
        ax = sns.barplot(data=table, x=table.columns[1], y="count", hue=obj_col, order=order)
  
        if len(order[0]) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
        
        for i in ax.containers:
          ax.bar_label(i,)
    else:
  
      fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
      x = 0
      y = 0
  
      for c in range(len(data.columns) - 1):
          
          if normalize:
            
        
            ax = axes[x,y]
            table = data.groupby(obj_col, as_index=False)[data.columns[c]].value_counts(normalize=normalize).round(2)
            order = sorted(table.iloc[:,1].unique())
            sns.barplot(data=table, x=table.columns[1], y="proportion", hue=obj_col, ax=ax, order=order)
  
            if len(order[0]) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
  
  
            if y != 0 and y%3 == 0:
                y = 0
                x += 1
            else:
                y += 1
          
            

          else:
  
            ax = axes[x,y]
            table = data.groupby(obj_col, as_index=False)[data.columns[c]].value_counts(normalize=normalize).round(2)
            order = sorted(table.iloc[:,1].unique())
            sns.barplot(data=table, x=table.columns[1], y="count", hue=obj_col, ax=ax, order=order)
  
            if len(order[0]) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
  
  
            if y != 0 and y%3 == 0:
                y = 0
                x += 1
            else:
                y += 1
        
  
    plt.tight_layout()

    
    
def fit_model(vars=None, year=None, clf="rf", sample=None, save=False, prefix=""):
  """
  Funcion para el entrenamiento de los modelos
  
  Parametros:
  --------------
      vars: Variables a utilizar para el modelo
      year: Año en que se tomaron los datos para entrenamiento
      clf: Tipo de algoritmo a utilizar. ("rf", "gb", "svc")
      sample: Usar estrategia de re-sampleo para el entrenamiento
      save: True si se quiere guardar el modelo como archivo. Default False
      prefix: prefijo a utilizar en el modelo, solo funciona si True es verdadero
      
  """  
    

  if year:
    query = f"SELECT * FROM df WHERE strftime('%Y', Date) = '{year}'"
  else:
    query = "SELECT * FROM df"

  df = create_df(query, index=True)

  target = "Attrition"

  if vars:
    X = df[vars]
  else:
    X = df.drop([target, "retirementType", "resignationReason", "Date"], axis=1)

  y = df[target]


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  if clf == "svc":
    cat_processor = OneHotEncoder()

  else:
    cat_processor = OrdinalEncoder()


  num_processor = StandardScaler()

  num_vals = X._get_numeric_data().columns.tolist()
  cat_vals = X.select_dtypes("object").columns.tolist()

  over_sampler = RandomOverSampler(random_state=42)
  under_sampler = RandomUnderSampler(random_state=42)

  X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
  X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

  processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

  if clf == "svc":

    model = make_pipeline(processor,
                      SVC(random_state=42))
  elif clf == "rf":

    model = make_pipeline(processor,
                      RandomForestClassifier(random_state=42))
  elif clf == "gb":

    model = make_pipeline(processor,
                      GradientBoostingClassifier(random_state=42))

  if sample == "over":
    model.fit(X_train_over, y_train_over)

  elif sample == "under":
    model.fit(X_train_under, y_train_under)

  else:
    model.fit(X_train, y_train)
  
  if os.path.exists('models'):
        pass
  else:
      os.makedirs(f"models/{prefix}/model_{prefix}")

  to_store = [model]

  for i in to_store:
      for name, value in locals().items():
          if i is value:
              if name == "i":
                  pass
  
              with open(f"models/{prefix}/{name}.pkl", "wb") as f:
                  pickle.dump(i, f)
      

  return model

def feature_sel(num_feat_kbest=20, num_rfe=15, plot_metric="chi2", plot=False, year_vars=None, sample=None, year_train=2016):
    """
    Funcion para la seleccion de las variables a utilizar
    
    Parametros:
    --------------
        num_feat_kbest: Variables a utilizar para de la seleccion por Kbest
        num_rfe: Variables a tener en cuenta con el metodo RFE
        plot_metric: Metrica por la cual se ordernaran los graficos
        plot: True si se quiere visualizar los resultados obtenidos
        year_vars: Año en que se tomaron los datos para entrenamiento de variables
        sample: Usar estrategia de re-sampleo para el entrenamiento
        year_train: Año en que se tomaron los datos para entrenamiento del modelo

    """  
      
    if year_vars:
        query = f"SELECT * FROM df WHERE strftime('%Y', Date) = '{year_vars}'"
    else:
        query = "SELECT * FROM df"

    df_vars = create_df(query, index=True)

    target = "Attrition"

    X = df_vars.drop([target, "retirementType", "resignationReason", "Date"], axis=1)
    y = df_vars[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_processor = OrdinalEncoder()
    num_processor = MinMaxScaler()

    num_vals = X._get_numeric_data().columns.tolist()
    cat_vals = X.select_dtypes("object").columns.tolist()

    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    vars = {"chi2":[], "mutual_info":[]}

    for m in (chi2, mutual_info_classif):
      for k in range(5, 25):

        selector = make_pipeline(processor,
                              SelectKBest(m, k=k))

        selector.fit(X_train, y_train)
        if m == chi2:
          vars["chi2"] += selector.get_feature_names_out().tolist()
        else:
          vars["mutual_info"] += selector.get_feature_names_out().tolist()


    vars_kb = pd.DataFrame({i:pd.Series(j).value_counts() for i,j in vars.items()})
    vars_kb.index = vars_kb.index.str[5:]

    if plot:

        vars_kb.sort_values(by=plot_metric, ascending=True).plot(kind="barh",)
        plt.legend(loc=[0.7, 0.2])
        plt.title("Feature Importance (chi2 vs mutual information)")
        plt.show()


    metric = ["chi2", "mutual_info"]
    criterion = ["gini", "entropy", "log_loss"]

    vars_rfe = pd.DataFrame()

    for m in metric:

      target = "Attrition"

      X = df_vars[vars_kb[m].sort_values(ascending=False).index.values.tolist()[:num_feat_kbest]]
      y = df_vars[target]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      cat_processor = OrdinalEncoder()
      num_processor = MinMaxScaler()

      num_vals = X._get_numeric_data().columns.tolist()
      cat_vals = X.select_dtypes("object").columns.tolist()

      processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

      for c in criterion:

        selector = make_pipeline(processor,
                                RFE(DecisionTreeClassifier(criterion=c, random_state=42), n_features_to_select=num_rfe))

        selector.fit(X_train, y_train)

        X.columns[selector.named_steps["rfe"].support_].values.tolist()

        vars_rfe[c+f"_{m}"] = X.columns[selector.named_steps["rfe"].support_].values



    counter = Counter()

    for i in vars_rfe.columns:
      counter.update(vars_rfe[i])


    n_select = pd.DataFrame(counter.values(), index=counter.keys()).rename(columns={0:"count"}).sort_values(by="count", ascending=True)

    if plot:
        n_select.plot(kind="barh", title="Numero de apariciones en los criterios", legend=False)
        plt.show()

    info = {i:[] for i in vars_rfe.columns}
    clfs = ["rf", "gb", "svc"]

    if year_train:
        query = f"SELECT * FROM df WHERE strftime('%Y', Date) = '{year_train}'"
    else:
        query = "SELECT * FROM df"

    df_train = create_df(query, index=True)

    for v in vars_rfe:
        for clf in clfs:
            vars = vars_rfe[v].values.tolist()
            target = "Attrition"
            X = df_train[vars]
            y = df_train[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = fit_model(vars=vars, clf=clf, sample=sample, year=year_train)
            pred = model.predict(X_test)
            info[v].append(f1_score(y_test, pred, pos_label="1"))

    _ = {"recurrente": []}
    for clf in clfs:
        vars = n_select.index.values.tolist()
        X = df_train[vars]
        y = df_train[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = fit_model(vars=vars, clf=clf, sample=sample, year=year_train)
        pred = model.predict(X_test)
        _["recurrente"].append(f1_score(y_test, pred, pos_label="1"))

    info.update(_)

    every = {"all_feat": []}
    for clf in clfs:
        vars = n_select.index.values.tolist()
        X = df_train.drop([target, "retirementType", "resignationReason", "Date"], axis=1)
        y = df_train[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = fit_model(vars=vars, clf=clf, sample=sample, year=year_train)
        pred = model.predict(X_test)
        every["all_feat"].append(f1_score(y_test, pred, pos_label="1"))

    info.update(every)
    
    eval_df = pd.DataFrame(info, index=[clfs])

    if os.path.exists('features'):
        pass
    else:
        os.mkdir("features")

    to_store = [vars_kb, vars_rfe, n_select, eval_df]

    for i in to_store:
      for name, value in locals().items():
        if i is value:
            if name == "i":
                pass

            with open(f"features/{name}.pkl", "wb") as f:
                 pickle.dump(i, f)    

    
def hyper_tunning(year=2016):
    
  """
  Funcion para el afinamiento de los hyperparametros del modelo seleccionado
  
  Parametros:
  --------------
      year_train: Año en que se tomaron los datos para entrenamiento del modelo
  """  

  with open("features/vars_rfe.pkl", "rb") as f:
      vars_rfe = pickle.load(f)

  param_grid = {"randomforestclassifier__n_estimators":range(25,100,25),
                "randomforestclassifier__criterion":["gini", "entropy", "log_loss"],
                "randomforestclassifier__max_features":["sqrt", "log2", None]}

  features = vars_rfe["gini_chi2"].values.tolist()

  if year:
      query = f"SELECT * FROM df WHERE strftime('%Y', Date) = '{year}'"
  else:
      query = "SELECT * FROM df"

  df = create_df(query, index=True)
  cv = 7

  target = "Attrition"

  X = df[features]
  y = df[target]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  cat_processor = OrdinalEncoder()
  num_processor = StandardScaler()

  num_vals = X._get_numeric_data().columns.tolist()
  cat_vals = X.select_dtypes("object").columns.tolist()

  processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

  clf = make_pipeline(processor,
                      RandomForestClassifier(random_state=42))

  model_tofit = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)


  model_tofit.fit(X_train, y_train)


  best_params = {i[len("randomforestclassifier__"):]:j for i,j in model_tofit.best_params_.items()}
  best_model = model_tofit.best_estimator_

  if os.path.exists('models/best_model'):
      pass
  else:
      os.makedirs("models/best_model")

  to_store = [features, best_params, df, best_model]

  for i in to_store:
    for name, value in locals().items():
      if i is value:
          if name == "i":
              pass

          with open(f"models/best_model/{name}.pkl", "wb") as f:
              pickle.dump(i, f)



def make_prediction(df=None):
    """
    Funcion para realizar la prediccion con nuevos datos
    
    Parametros:
    ---------------
        df: Data frame con la informacion a predecir
    """

    assert isinstance(df, pd.DataFrame), "Se necesita una base de datos para la prediccion"

    with open("models/best_model/features.pkl", "rb") as f:
        feat = pickle.load(f)

    with open("models/best_model/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    X = df[feat]

    prediction = model.predict(X)

    pred_df = pd.DataFrame(prediction, index=X.index, columns=["Prediccion"])

    pred_df.to_csv("models/best_model/predictions.csv", index=False)

    return prediction   