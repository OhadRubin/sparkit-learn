import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"
import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
# from pyspark.mllib.feature import HashingTF, IDF

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
conf = SparkConf()
conf.set("spark.app.name", "MinHashLSH")
conf.set("spark.debug.maxToStringFields", "100")
conf.set("spark.local.dir", "/dev/shm/pyspark_dir") #TODO: move in arguements
conf.set("spark.driver.memory", "64g")
conf.set("spark.executor.memory", "64g")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


from splearn.rdd import ArrayRDD
from splearn.feature_extraction.text import SparkCountVectorizer, SparkHashingVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer
from splearn.decomposition import SparkTruncatedSVD
from splearn.pipeline import SparkPipeline

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import pandas as pd
X = pd.read_csv("/home/ohadr/database_project_c/test_data/sample.csv").text.tolist()

X_rdd = ArrayRDD(spark.sparkContext.parallelize(X, 4))  # Get SparkContext from SparkSession

# Use CountVectorizer as requested
vectorizer = CountVectorizer()
spark_vectorizer = SparkCountVectorizer()

local_pipeline = Pipeline((
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('pca', TruncatedSVD(n_components=2))
))
dist_pipeline = SparkPipeline((
    ('vect', spark_vectorizer),
    ('tfidf', SparkTfidfTransformer()),
    ('pca', SparkTruncatedSVD(n_components=2))
))

result_local = local_pipeline.fit_transform(X)
result_dist = dist_pipeline.fit_transform(X_rdd)  # SparseRDD

print(result_dist.collect())
print("Successfully completed regression test!")