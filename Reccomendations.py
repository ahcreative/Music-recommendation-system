from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MusicRecommendation") \
    .config("spark.mongodb.input.uri", "mongodb://192.168.93.129 :27017/db.collection") \
    .getOrCreate()

# Load data from MongoDB into Spark DataFrame
df = spark.read.format("mongo").load()

from pyspark.ml.feature import StringIndexer

# Convert categorical variables to numerical values
indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
indexed_df = indexer.fit(df).transform(df)

# Split data into training and testing sets
(training_data, testing_data) = indexed_df.randomSplit([0.8, 0.2])

from pyspark.ml.recommendation import ALS

# Initialize ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="song_id", ratingCol="rating",
          coldStartStrategy="drop")

# Train ALS model
model = als.fit(training_data)

from pyspark.ml.evaluation import RegressionEvaluator

# Make predictions
predictions = model.transform(testing_data)

# Evaluate model using RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = " + str(rmse))
