from flask import Flask, jsonify
from kafka import KafkaProducer, KafkaConsumer

app = Flask(_name_)

# Initialize Kafka producer and consumer
producer = KafkaProducer(bootstrap_servers='192.168.93.129 :9092')
consumer = KafkaConsumer('user_activity', bootstrap_servers='192.168.93.129 :9092', group_id='music_streaming_group')

# Define routes
@app.route('/')
def index():
    return "Welcome to Music Streaming Service!"

@app.route('/recommendations/<user_id>')
def get_recommendations(user_id):
    # Send user activity to Kafka
    producer.send('user_activity', key=user_id.encode(), value=b'play')

    # Consume recommendation from Kafka
    recommendations = []
    for message in consumer:
        recommendations.append(message.value.decode())
        if len(recommendations) == 5:
            break
    
    return jsonify(recommendations)

if _name_ == '_main_':
    app.run(debug=True)
