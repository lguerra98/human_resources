import sqlite3
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt



def create_db(path=r"C:\Users\USUARIO\OneDrive - Universidad de Antioquia\Analitica_3"):
    """
    Funcion para la creacion del Database

    Parametros
    ------------
        path: direccion local de los archivos como una raw string

    """
    
    if os.path.exists("data/human_db"):
        
        os.unlink("data/human_db")
        os.rmdir("data")
        
        
    # if os.path.exists("data/human_db"):
    #     pass
    
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
        cursor.execute("DELETE FROM retirement_info WHERE retirementType = 'Fired'")
        
        ## Crear nueva tabla
        cursor.execute("CREATE TABLE IF NOT EXISTS general_data_up AS SELECT EmployeeID, InfoDate AS Date, Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, Gender, JobLevel, JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager	FROM general_data")
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
    
        ## Eliminar tablas modificadas
        # cursor.execute("DROP TABLE IF EXISTS general_data")
        # cursor.execute("DROP TABLE IF EXISTS employee_survey_data")
        # cursor.execute("DROP TABLE IF EXISTS manager_survey")
        # cursor.execute("DROP TABLE IF EXISTS retirement_info")
    
        ## Cambio de catiegorias con muy poca influencia en las variables
        cursor.execute("UPDATE manager_survey_up SET JobInvolvement = 0 WHERE JobInvolvement IN (1, 2)")
        cursor.execute("UPDATE manager_survey_up SET JobInvolvement = 1 WHERE JobInvolvement IN (3, 4)")
        cursor.execute("UPDATE general_data_up SET Education = 4 WHERE Education = 5")
        
        ## Eliminar datos del 2015 ya que son iguales y de poca relevancia
        cursor.execute("DELETE FROM general_data_up WHERE strftime('%Y', Date) = '2015'")
        cursor.execute("DELETE FROM manager_survey_up WHERE strftime('%Y', Date) = '2015'")
        cursor.execute("DELETE FROM retirement_info_up WHERE strftime('%Y', Date) = '2015'")
        cursor.execute("DELETE FROM employee_survey_data_up WHERE strftime('%Y', Date) = '2015'")
            
        ## Crear tabla con toda la informacion
        cursor.execute("CREATE TABLE IF NOT EXISTS df AS SELECT EmployeeID, Date, CAST(EnvironmentSatisfaction AS TEXT) AS EnvironmentSatisfaction, CAST(JobSatisfaction AS TEXT) AS JobSatisfaction, CAST(WorkLifeBalance AS TEXT) AS WorkLifeBalance, Age, BusinessTravel, Department, DistanceFromHome, CAST(Education AS TEXT) Education, EducationField, Gender, CAST(JobLevel AS TEXT) JobLevel, CAST(JobRole AS TEXT) JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, CAST(StockOptionLevel AS TEXT) AS StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, CAST(JobInvolvement AS TEXT) AS JobInvolvement, CAST(PerformanceRating AS TEXT) AS PerformanceRating, retirementType, resignationReason, Attrition FROM (SELECT * FROM employee_survey_data_up INNER JOIN general_data_up ON employee_survey_data_up.EmployeeID = general_data_up.EmployeeID INNER JOIN manager_survey_up ON employee_survey_data_up.EmployeeID = manager_survey_up.EmployeeID LEFT JOIN retirement_info_up ON employee_survey_data_up.EmployeeID = retirement_info_up.EmployeeID)")
        ## Llenar valores nulos de la nueva tabla
        cursor.execute("UPDATE df SET retirementType = 'Active', resignationReason = 'Not applicable' WHERE retirementType IS NULL")
        cursor.execute("UPDATE df SET Attrition = '0' WHERE Attrition IS NULL")
        cursor.execute("UPDATE df SET Attrition = '1' WHERE Attrition = 'Yes'")
        
        
        conn.commit()
        
        
        
            
        conn.close()
    
    

def create_df(query, index=False):

    conn = sqlite3.connect("data/human_db")

    if index:
        df = pd.read_sql(query, conn, index_col="EmployeeID")
    else:
        df = pd.read_sql(query, conn)
    conn.close()
    return df

df = create_df("SELECT * FROM df", index=True)


def heatmap(df=df, table=False):
    corr = df._get_numeric_data().corr()
    
    if table:
        

        sns.heatmap(corr.abs());
        
        display(corr.round(2).style.background_gradient(axis=None))
    else:
        sns.heatmap(corr.abs());
        
        
        
def pieplot(df=df, groupby="Attrition", count_col="Date"):

    table = df.groupby([groupby], as_index=False)[count_col].value_counts().drop(columns=count_col)
    
    table.plot(kind="pie", y="count", autopct='%1.1f%%', 
               labels=["Stayed", "Left"], ylabel="", 
               explode=(0.13, 0), shadow=True, labeldistance=.5, pctdistance=1.3);
    

    
    
def histograms(df=df, nrows=3, ncols=3, figsize=(12, 12), var_obj="Attrition"):

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
   


def barcharts(df=df, normalize=True, obj_col="Attrition", rotation=90, only=False, targ_col=...):


  data = df.select_dtypes(include=["object"]).drop(columns="Date")


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
          
          if data.columns[c] == "retirementType":
              break
          
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