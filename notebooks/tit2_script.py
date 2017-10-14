# coding: utf-8

# In[2]:


import pyspark

sc = pyspark.SparkContext('local[*]')


from pyspark.sql import SQLContext

sqlc = SQLContext(sc)

# ### Step 1
# - Load the train and test sets
# - Check the schema, the variables have their right types?
# - If not, how to correctly load the datasets?

# In[3]:


from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

titanicSchemaTrain = StructType([StructField("PassengerId", IntegerType(), True),
                                 StructField("Survived", IntegerType(), True),
                                 StructField("Pclass", IntegerType(), True),
                                 StructField("Name", StringType(), True),
                                 StructField("Sex", StringType(), True),
                                 StructField("Age", FloatType(), True),
                                 StructField("SibSp", IntegerType(), True),
                                 StructField("Parch", IntegerType(), True),
                                 StructField("Ticket", StringType(), True),
                                 StructField("Fare", FloatType(), True),
                                 StructField("Cabin", StringType(), True),
                                 StructField("Embarked", StringType(), True)]
                                )

titanicSchemaTest = StructType([StructField("PassengerId", IntegerType(), True),
                                StructField("Pclass", IntegerType(), True),
                                StructField("Name", StringType(), True),
                                StructField("Sex", StringType(), True),
                                StructField("Age", FloatType(), True),
                                StructField("SibSp", IntegerType(), True),
                                StructField("Parch", IntegerType(), True),
                                StructField("Ticket", StringType(), True),
                                StructField("Fare", FloatType(), True),
                                StructField("Cabin", StringType(), True),
                                StructField("Embarked", StringType(), True)]
                               )
df_train = sqlc.read.load(path="data/train.csv",
                          format="com.databricks.spark.csv",
                          schema=titanicSchemaTrain,
                          header=True)

df_test = sqlc.read.load(path="data/test.csv",
                         format="com.databricks.spark.csv",
                         schema=titanicSchemaTest, header=True)

# ### Step 2
# - Explore the features of your dataset
# - You can use DataFrame's ***describe*** method to get summary statistics
#     - hint: ***toPandas*** may be useful to ease the manipulation of small dataframes
# - Are there any ***NaN*** values in your dataset?
# - If so, define value/values to fill these ***NaN*** values
#     - hint: ***na*** property of DataFrames provide several methods of handling NA values

# ### Step 3
# - How to handle categorical features?
#     - hint: check the Estimators and Transformers
# - Assemble all desired features into a Vector using the VectorAssembler Transformer
# - Make sure to end up with a DataFrame with two columns: ***Survived*** and ***vFeatures***

# In[4]:


age_mean = df_train.describe().toPandas().set_index("summary").loc['mean', 'Age']


def remove_useless_features(df):
    return df.drop("Cabin")


df_train = remove_useless_features(df_train)
df_test = remove_useless_features(df_test)
from numpy import NaN
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import isnull, isnan, when, count, col


def average_missing_features(df):
    df = df.withColumn("age", when(col('age').isNull(), age_mean).otherwise(col('age')))
    df = df.withColumn("Embarked", when(col('Embarked').isNull(), 'C').otherwise(col('Embarked')))
    df = df.withColumn("Fare", when(col('Fare').isNull(), 0).otherwise(col('Fare')))

    return df


df_train = average_missing_features(df_train)
df_test = average_missing_features(df_test)

# In[7]:


from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType


def convert_age(df):
    df = df.withColumn("age", col("age").cast(FloatType()))
    return df


df_train = convert_age(df_train)
df_test = convert_age(df_test)

# In[71]:


df_train


# In[72]:




# extract_title(df_train)


# In[73]:


df_test.describe().toPandas()

# Step 3
# How to handle categorical features?
# hint: check the Estimators and Transformers
# Assemble all desired features into a Vector using the VectorAssembler Transformer
# Make sure to end up with a DataFrame with two columns: Survived and vFeatures
#

# In[74]:


from pyspark.ml.feature import StringIndexer


def categorize_df(df):
    indexerS = StringIndexer(inputCol="Sex", outputCol="SexC")
    indexerE = StringIndexer(inputCol="Embarked", outputCol="EmbarkedC")

    df = indexerS.fit(df).transform(df)
    df = indexerE.fit(df).transform(df)

    df = df.drop("Sex", "Embarked")
    return df


#df_train = categorize_df(df_train)
#df_test = categorize_df(df_test)

# In[75]:


from pyspark.ml.feature import OneHotEncoder


def onehot_df(df):
    oneHotS = OneHotEncoder(inputCol="SexC", outputCol="SexV")
    oneHotE = OneHotEncoder(inputCol="EmbarkedC", outputCol="EmbarkedV")

    df = oneHotS.transform(df)
    df = oneHotE.transform(df)

    df = df.drop("SexC", "EmbarkedC")
    return df


#df_train = onehot_df(df_train)
#df_test = onehot_df(df_test)

# In[76]:






# In[77]:


from pyspark.ml.feature import VectorAssembler


def vectorize(df):
    assembler = VectorAssembler(inputCols=["Pclass", "age", "SibSp", "Parch", "Fare", "SexV", "EmbarkedV"],
                                outputCol="vFeatures")

    df = assembler.transform(df)

    return df


#df_train = vectorize(df_train)
#df_test = vectorize(df_test)

#df_train = df_train['PassengerId', 'vFeatures', 'Survived']
#df_test = df_test['PassengerId', 'vFeatures']

# In[78]:


df_train.toPandas()

# In[58]:


df_test.toPandas()

# In[ ]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils

### INSERT YOUR CODE HERE


# ### Step 4
# - In Step 5, you will apply a normalization Estimator
# - BUT, it does not accept feature vectors of the Sparse type
# - So, it is neccessary to apply an User Defined Function to make all features vectors of type VectorUDT
# - In this step, you only have to replace ***YOUR DATAFRAME*** and ***NEW DATAFRAME*** with your variables

# In[79]:


from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.linalg import VectorUDT, Vectors

#to_vec = UserDefinedFunction(lambda x: Vectors.dense(x.toArray()), VectorUDT())

#df_train = df_train.select("PassengerId", "Survived", to_vec("vFeatures").alias("features"))
#df_test = df_test.select("PassengerId", to_vec("vFeatures").alias("features"))

# ### Step 5
# - Apply a normalization Estimator of your choice to the ***features*** vector obtained in Step 4

# In[80]:


from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
#df_train_scalerModel = scaler.fit(df_train)

# Normalize each feature to have unit standard deviation.
#df_train = df_train_scalerModel.transform(df_train)

# Compute summary statistics by fitting the StandardScaler
#df_test_scalerModel = scaler.fit(df_test)

# Normalize each feature to have unit standard deviation.
#df_test = df_test_scalerModel.transform(df_test)

# In[82]:


df_train.toPandas()

# ### Step 6
# - Train a classifier of your choice (for instance, Random Forest) using your dataset of LabeledPoints
# - Make predictions for the training data
# - Use the Binary Classification Evaluator to evaluate your model on the training data
# - How is your model performing? Try to tune its parameters

# In[16]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import SQLTransformer

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="Sex", outputCol="SexC"),
    StringIndexer(inputCol="Embarked", outputCol="EmbarkedC"),
    OneHotEncoder(inputCol="SexC", outputCol="SexV"),
    OneHotEncoder(inputCol="EmbarkedC", outputCol="EmbarkedV"),
    VectorAssembler(inputCols= \
                        ["Pclass", "age", "SibSp", "Parch", "Fare", "SexV", "EmbarkedV"], outputCol="features"),
    StandardScaler(inputCol="features", outputCol="scaledFeatures",
                   withStd=True, withMean=False),
    SQLTransformer(statement="SELECT PassengerId, Survived, scaledFeatures FROM __THIS__")
])

# Fit the pipeline to training documents.
model = pipeline.fit(df_train)
df_train_2 = model.transform(df_train)
df_train_3 = df_train_2

# In[17]:


df_train_3.show()

# In[18]:


df_train_3.select('Survived', 'scaledFeatures').toPandas()

# In[ ]:




# ## Result = ???%