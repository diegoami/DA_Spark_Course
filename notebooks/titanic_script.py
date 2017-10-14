from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import SQLTransformer

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

import pyspark
sc = pyspark.SparkContext('local[*]')
from pyspark.sql import SQLContext
sqlc = SQLContext(sc)

titanicSchemaTrain = StructType([StructField("PassengerId", IntegerType(), True),
                           StructField("Survived", IntegerType(), True),
                           StructField("Pclass",  IntegerType(), True),
                           StructField("Name",  StringType(), True),
                           StructField("Sex",  StringType(), True),
                           StructField("Age",  FloatType(), True),
                           StructField("SibSp",  IntegerType(), True),
                           StructField("Parch",  IntegerType(), True),
                           StructField("Ticket",  StringType(), True),
                            StructField("Fare",  FloatType(), True),
                            StructField("Cabin",  StringType(), True),
                           StructField("Embarked",  StringType(), True)]
                          )

titanicSchemaTest = StructType([StructField("PassengerId", IntegerType(), True),
                           StructField("Pclass",  IntegerType(), True),
                           StructField("Name",  StringType(), True),
                           StructField("Sex",  StringType(), True),
                           StructField("Age",  FloatType(), True),
                           StructField("SibSp",  IntegerType(), True),
                           StructField("Parch",  IntegerType(), True),
                           StructField("Ticket",  StringType(), True),
                            StructField("Fare",  FloatType(), True),
                            StructField("Cabin",  StringType(), True),
                               StructField("Embarked",  StringType(), True)]
                          )
df_train = sqlc.read.load(path="data/train.csv",
                          format="com.databricks.spark.csv",
                          schema=titanicSchemaTrain,
                          header=True)

df_test = sqlc.read.load(path="data/test.csv",
                          format="com.databricks.spark.csv",
                          schema=titanicSchemaTest, header=True)

age_mean = df_train.describe().toPandas().set_index("summary").loc['mean', 'Age']


def remove_useless_features(df):
    return df.drop("Cabin")

print(df_train.describe())

df_train = remove_useless_features(df_train)
df_test = remove_useless_features(df_test)

print(df_train.describe())
from numpy import NaN
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import isnull, isnan, when, col

print(df_train.describe())
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
def convert_age(df):
    df = df.withColumn("age", col("age").cast(FloatType()))
    return df
df_train = convert_age(df_train)
df_test = convert_age(df_test)


print(df_train.describe())

def average_missing_features(df):
    df = df.withColumn("age", when(col('age').isNull(), age_mean).otherwise(col('age')))
    df = df.withColumn("Embarked", when(col('Embarked').isNull(), 'C').otherwise(col('Embarked')))
    df = df.withColumn("Fare", when(col('Fare').isNull(), 0).otherwise(col('Fare')))

    return df


df_train = average_missing_features(df_train)
df_train.show()

df_test = average_missing_features(df_test)

print(df_train.describe())
pipeline = Pipeline(stages=[
    StringIndexer(inputCol="Sex", outputCol="SexC"),
    StringIndexer(inputCol="Embarked", outputCol="EmbarkedC"),
    OneHotEncoder(inputCol="SexC", outputCol="SexV"),
    OneHotEncoder(inputCol="EmbarkedC", outputCol="EmbarkedV"),
    VectorAssembler(inputCols = \
                ["Pclass","age", "SibSp", "Parch", "Fare", "SexV", "EmbarkedV"], outputCol = "features"),
    StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False),
    SQLTransformer(statement="SELECT PassengerId, Survived, scaledFeatures FROM __THIS__")
])
# Fit the pipeline to training documents.
model = pipeline.fit(df_train)
df_train_2 =  model.transform(df_train)
df_train_3 = df_train_2
df_train_3.show()