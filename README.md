# Music-recommendation-system
This project is a music streaming service that utilizes various technologies to provide personalized music recommendations to users. Here's a detailed explanation of the project:

1. **Flask Backend (`Flask_backend.py`)**
   - This file contains a Flask application that serves as the backend for the music streaming service.
   - It initializes a Kafka producer and consumer to handle user activity and receive recommendations, respectively.
   - The backend exposes two routes:
     - `/`: The root route that displays a welcome message.
     - `/recommendations/<user_id>`: This route is responsible for handling user requests for music recommendations. When a user requests recommendations, the backend sends the user's activity (in this case, 'play') to the Kafka topic 'user_activity'. It then consumes and returns the top 5 recommendations received from the Kafka topic.

2. **Flask Frontend (`Flask_Frontend.html`)**
   - This file contains the HTML code for the frontend interface of the music streaming service.
   - It provides a form where users can enter their user ID and submit a request to get music recommendations.
   - The frontend uses JavaScript to fetch recommendations from the backend's `/recommendations/<user_id>` route and display them on the page.

3. **MongoDB Integration (`Mongodb.py`)**
   - This Python script is responsible for loading audio features and metadata into a MongoDB database.
   - It extracts audio features (MFCC, spectral centroid, and zero-crossing rate) from audio files using the librosa library.
   - The extracted features are preprocessed by standardization and dimensionality reduction using PCA.
   - The script reads and preprocesses metadata files (albums, artists, genres, and tracks) from a provided ZIP archive or directory.
   - The preprocessed features and metadata are then loaded into a MongoDB database for efficient storage and retrieval.

4. **Recommendations Engine (`Reccomendations.py`)**
   - This Python script leverages PySpark to build a music recommendation system using collaborative filtering.
   - It initializes a Spark session and loads data from the MongoDB database into a Spark DataFrame.
   - Categorical variables (e.g., genre) are indexed for numerical representation.
   - The data is split into training and testing sets.
   - An Alternating Least Squares (ALS) model is trained on the training data for collaborative filtering.
   - The trained model is used to make predictions on the testing data.
   - The Root Mean Squared Error (RMSE) is calculated to evaluate the model's performance.

5. **Data Sampling (`Sampling.ipynb`)**
   - This Jupyter Notebook contains code for sampling a subset of audio files from a larger ZIP archive.
   - It defines a function `extract_subset` that takes a ZIP file path, an output folder, a maximum file count, and a maximum total size (in GB) as input.
   - The function extracts files from the ZIP archive into the output folder, stopping when either the file count or total size limit is reached.
   - This allows for efficient processing of a smaller, representative sample of the audio data.

Overall, this project combines various components to create a music streaming service with personalized recommendations. It leverages Flask for the backend and frontend, Kafka for messaging, MongoDB for data storage, and PySpark for building the recommendation engine using collaborative filtering. The data sampling notebook helps in working with a manageable subset of the audio data during development and testing.
