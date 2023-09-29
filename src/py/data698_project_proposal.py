# Databricks notebook source
# MAGIC %md
# MAGIC ### CUNY SPS DATA 698 Project Proposal
# MAGIC
# MAGIC #### Name: Chinedu Onyeka
# MAGIC #### Date: September 30th, 2023

# COMMAND ----------

spark.conf.set("spark.databricks.delta.formatCheck.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Libraries

# COMMAND ----------

# libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data

# COMMAND ----------

# variables
train_raw_path = "https://raw.githubusercontent.com/chinedu2301/data698-analytics-project/main/data/vehicle_default_train_data.csv"

# COMMAND ----------

# read file from github
vehicle_default_train_raw = pd.read_csv(train_raw_path)
vehicle_default_train_raw_df = spark.createDataFrame(vehicle_default_train_raw)

# COMMAND ----------

display(vehicle_default_train_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the data will need some pre-processing and cleaning.

# COMMAND ----------

# view the columns and data type for the data
vehicle_default_train_raw.info()

# COMMAND ----------

vehicle_default_train_raw.shape

# COMMAND ----------

# MAGIC %md
# MAGIC There are 41 columns and 233,154 rows in the data. There are 40 predictor variables and 1 response variable. The response variable is binary (1, 0) where 1 indicate a loan default and 0 indicate no-loan default. The data dictionary for the data can be found [here](https://github.com/chinedu2301/data698-analytics-project/blob/main/data/data_dictionary.xlsx)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Research Question
# MAGIC This project aims to develop a machine learning model that will predict whether a borrower will default or not. Being able to determine whether a borrower will default on their auto-loan would help the business determine whether to extend credit or not thereby helping the business to minimize losses due to a borrower not being able to meet up with their auto-payments. Also, it would help the business to extend credit/loan to those who would not default. This way, the business can be sure to a certain degree that customers who apply for loan get the right decision that will be beneficial to the business.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Source
# MAGIC The data was gotten from kaggle and it's available in my github account here for the [train](https://github.com/chinedu2301/data698-analytics-project/blob/main/data/vehicle_default_train_data.csv) and [test](https://github.com/chinedu2301/data698-analytics-project/blob/main/data/vehicle_default_test_data.csv) data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Relevant Statistics

# COMMAND ----------

vehicle_default_train_raw.describe()

# COMMAND ----------

vehicle_default_train_raw["LOAN_DEFAULT"].value_counts()

# COMMAND ----------

# Plot Loan Default Category counts
loan_default_counts  = vehicle_default_train_raw["LOAN_DEFAULT"].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(loan_default_counts.index, loan_default_counts.values, alpha=0.8)
plt.title('Count for Loan Default Category')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Loan Default', fontsize=12)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that there are way more records for "0" and we might need to do oversampling of the "1" rows to balance the data before training the model.
