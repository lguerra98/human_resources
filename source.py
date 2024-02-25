import sqlite3
import pandas as pd
import os



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
        
        # os.unlink("data/human_db")
        # os.rmdir("data")
        
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
        cursor.execute("CREATE TABLE IF NOT EXISTS retirement_info_up AS SELECT EmployeeID, retirementDate AS Date, retirementType, resignationReason FROM retirement_info")
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
        cursor.execute("CREATE TABLE IF NOT EXISTS df AS SELECT EmployeeID, Date, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance, Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, Gender, JobLevel, JobRole, MaritalStatus, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, JobInvolvement, PerformanceRating, retirementType, resignationReason FROM (SELECT * FROM employee_survey_data_up INNER JOIN general_data_up ON employee_survey_data_up.EmployeeID = general_data_up.EmployeeID INNER JOIN manager_survey_up ON employee_survey_data_up.EmployeeID = manager_survey_up.EmployeeID LEFT JOIN retirement_info_up ON employee_survey_data_up.EmployeeID = retirement_info_up.EmployeeID);")
        ## Llenar valores nulos de la nueva tabla
        cursor.execute("UPDATE df SET retirementType = 'Active' WHERE retirementType IS NULL")
        cursor.execute("UPDATE df SET resignationReason = 'Not applicable' WHERE resignationReason IS NULL")
        
        
        
        conn.commit()
        
        
        
     
        
        
        
            
        conn.close()