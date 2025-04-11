import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("clipper-pyspark").getOrCreate()

spark_context = spark.sparkContext

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(60)

# Read CSV with Pandas 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Pandas Train Data (Head):")
print(train_data.head())

# - Read CSV into Spark DataFrame 
from pyspark.sql.functions import col, lower

raw_df = spark.read.format('csv') \
    .option('header', 'true') \
    .option('inferSchema', 'true') \
    .option('timestamp', 'true') \
    .load('train.csv')

spark_data = (
    raw_df.select(lower(col('Category')).alias('Category'),
                  lower(col('Descript')).alias('Description'))
)
spark_data.cache()

print("\nDataFrame Structure")
print("----------------------------------")
spark_data.printSchema()

print("\nDataFrame Preview")
spark_data.show(5)

print("----------------------------------")
print("Total number of rows:", raw_df.count())

# Utility Function 
def get_top_n_values(dataframe, col_name, n):
    """
    Prints the total distinct count of `col_name` and shows the top `n` most frequent values.
    """
    unique_count = dataframe.select(col_name).distinct().count()
    print(f"Total unique values in '{col_name}': {unique_count}\n")
    print(f"Top {n} Values in column '{col_name}':")
    dataframe.groupBy(col_name) \
             .count() \
             .withColumnRenamed('count', 'totalValue') \
             .orderBy(col('totalValue').desc()) \
             .show(n)

# Check Top Values 
get_top_n_values(spark_data, 'Category', 10)
print()
get_top_n_values(spark_data, 'Description', 10)

#  Split Data 
training_data, test_data = spark_data.randomSplit([0.7, 0.3], seed=60)
print(f"\nTraining Dataset Count: {training_data.count()}")
print(f"Test Dataset Count: {test_data.count()}")

# Feature Engineering & ML Imports 
from pyspark.ml.feature import (
    RegexTokenizer, 
    StopWordsRemover, 
    CountVectorizer, 
    StringIndexer, 
    HashingTF, 
    IDF, 
    Word2Vec
)
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Tokenizer
regex_tokenizer = RegexTokenizer(pattern='\\W', inputCol="Description", outputCol="tokens")

# StopWords
extra_stopwords = ['http','amp','rt','t','c','the']
stopwords_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered_words', stopWords=extra_stopwords)

# CountVectorizer
count_vectors = CountVectorizer(vocabSize=10000, minDF=5, inputCol="filtered_words", outputCol="features")

# TF-IDF
hashingTf = HashingTF(numFeatures=10000, inputCol="filtered_words", outputCol="raw_features")
idf = IDF(minDocFreq=5, inputCol="raw_features", outputCol="features")

# Word2Vec
word2Vec = Word2Vec(vectorSize=1000, minCount=0, inputCol="filtered_words", outputCol="features")

# Encode Labels
label_string_idx = StringIndexer(inputCol="Category", outputCol="label")

# Classifiers
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
nb = NaiveBayes(smoothing=1)

# Helper: Metrics Evaluation 
def evaluate_metrics(class_labels, eval_metrics):
    """
    Prints confusion matrix & performance metrics for classification.
    """
    print("--------- Confusion Matrix ---------")
    print(eval_metrics.confusionMatrix)
    print("\n--------- Overall Statistics ---------")
    print(f"Precision: {eval_metrics.precision()}")
    print(f"Recall:    {eval_metrics.recall()}")
    print(f"F1 Score:  {eval_metrics.fMeasure()}\n")

    print("------ Statistics by Class ------")
    for label in sorted(class_labels):
        print(f"Class {label} precision = {eval_metrics.precision(label)}")
        print(f"Class {label} recall    = {eval_metrics.recall(label)}")
        print(f"Class {label} F1 Score  = {eval_metrics.fMeasure(label, beta=1.0)}\n")

    print("------ Weighted Statistics ------")
    print(f"Weighted Recall               = {eval_metrics.weightedRecall}")
    print(f"Weighted Precision            = {eval_metrics.weightedPrecision}")
    print(f"Weighted F(1) Score           = {eval_metrics.weightedFMeasure()}")
    print(f"Weighted F(0.5) Score         = {eval_metrics.weightedFMeasure(beta=0.5)}")
    print(f"Weighted False Positive Rate  = {eval_metrics.weightedFalsePositiveRate}\n")

#  Pipelines & Models
# CountVectorizer + LR
pipeline_cv_lr = Pipeline(stages=[regex_tokenizer, stopwords_remover, count_vectors, label_string_idx, lr])
model_cv_lr = pipeline_cv_lr.fit(training_data)
predictions_cv_lr = model_cv_lr.transform(test_data)

print("\n-- Top 5 Predictions (CountVectorizer + LogisticRegression) --")
predictions_cv_lr.select("Description", "Category", "probability", "label", "prediction") \
                 .orderBy("probability", ascending=False) \
                 .show(5, truncate=30)

evaluator_cv_lr = MulticlassClassificationEvaluator(predictionCol="prediction").evaluate(predictions_cv_lr)
print(f"\nAccuracy (CV + LR): {evaluator_cv_lr}")

# CountVectorizer + NB
pipeline_cv_nb = Pipeline(stages=[regex_tokenizer, stopwords_remover, count_vectors, label_string_idx, nb])
model_cv_nb = pipeline_cv_nb.fit(training_data)
predictions_cv_nb = model_cv_nb.transform(test_data)
evaluator_cv_nb = MulticlassClassificationEvaluator(predictionCol="prediction").evaluate(predictions_cv_nb)
print(f"\nAccuracy (CV + NB): {evaluator_cv_nb}")

# TF-IDF + LR
pipeline_idf_lr = Pipeline(stages=[regex_tokenizer, stopwords_remover, hashingTf, idf, label_string_idx, lr])
model_idf_lr = pipeline_idf_lr.fit(training_data)
predictions_idf_lr = model_idf_lr.transform(test_data)

print("\n-- Top 5 Predictions (TF-IDF + LogisticRegression) --")
predictions_idf_lr.select("Description", "Category", "probability", "label", "prediction") \
                  .orderBy("probability", ascending=False) \
                  .show(5, truncate=30)

evaluator_idf_lr = MulticlassClassificationEvaluator(predictionCol="prediction").evaluate(predictions_idf_lr)
print(f"\nAccuracy (TF-IDF + LR): {evaluator_idf_lr}")

# TF-IDF + NB
pipeline_idf_nb = Pipeline(stages=[regex_tokenizer, stopwords_remover, hashingTf, idf, label_string_idx, nb])
model_idf_nb = pipeline_idf_nb.fit(training_data)
predictions_idf_nb = model_idf_nb.transform(test_data)
evaluator_idf_nb = MulticlassClassificationEvaluator(predictionCol="prediction").evaluate(predictions_idf_nb)
print(f"\nAccuracy (TF-IDF + NB): {evaluator_idf_nb}")

# Word2Vec + LR
pipeline_wv_lr = Pipeline(stages=[regex_tokenizer, stopwords_remover, word2Vec, label_string_idx, lr])
model_wv_lr = pipeline_wv_lr.fit(training_data)
predictions_wv_lr = model_wv_lr.transform(test_data)
evaluator_wv_lr = MulticlassClassificationEvaluator(predictionCol="prediction").evaluate(predictions_wv_lr)
print("\nAccuracy (Word2Vec + LR):", evaluator_wv_lr)
