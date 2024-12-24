import joblib

# Load the model and vectorizer
model = joblib.load('/content/mnb_model.pkl')
tfidf = joblib.load('/content/tfidf_vectorizer.pkl')

def predictReview(review: str) -> str:
    """
    Given a review text, predicts the sentiment using a pre-trained model.

    :param review: The review text (string).
    :return: The predicted sentiment (string).
    """
    # Vectorize the review text and predict sentiment
    review_vector = tfidf.transform([review])
    prediction = model.predict(review_vector)[0]  # Assuming binary or multi-class prediction

    return str(prediction)
from google.colab import files
import os

# Upload the image
uploaded = files.upload()

# Create a static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Move the uploaded image to the static directory
for filename in uploaded.keys():
    os.rename(filename, f'static/{filename}')
!pip install flask pyngrok
from google.colab import userdata
ngrok_token = userdata.get('ngrok_token')
# !ngrok config add-authtoken ngrok_token
!ngrok config add-authtoken "YOUR NGROK TOKEN"
from flask import Flask, render_template_string, request, jsonify
from pyngrok import ngrok
import json

# Initialize the Flask app
app = Flask(__name__)

# HTML content with inline CSS and JavaScript
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis of meesho's reviews using Multinomial naive bayes model</h1>
        <form action="{{ url_for('predict')}}" method="post"></form>
        <textarea id="reviewInput" placeholder="Enter your review here..."></textarea>
        <button onclick="predictSentiment()">Analyze Sentiment</button>

        <div id="result">
            <h3>Prediction: <span id="prediction"></span></h3>
        </div>
    </div>

    <script>
    function predictSentiment() {
        const review = document.getElementById('reviewInput').value;

        // Send the input to Flask backend using POST request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: review }),
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction and confidence scores
            document.getElementById('prediction').innerText = data.prediction;
        })
        .catch(error => console.error('Error:', error));
    }
    </script>
</body>
</html>
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    background-image: url('/static/Background.jpg'); /* Add this line with your image file path */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.container {
    text-align: center;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

textarea {
    width: 80%;
    height: 100px;
    margin-bottom: 10px;
}

button {
    padding: 10px 20px;
    margin-bottom: 20px;
    cursor: pointer;
}

#result {
    margin-top: 10px;
}
</style>
"""

# Route to render HTML page
@app.route('/')
def index():
    return render_template_string(html_content)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
        data = request.get_json()  # This converts the JSON data into a Python dictionary

        # Get the 'review' field from the JSON data
        review = data.get('review', '')

        # If 'review' is not found or empty, return an error message
        if not review:
            return jsonify({"error": "No review provided"}), 400

        # Call the prediction function (assume you already have the function defined)
        prediction = predictReview(review)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction})


# Start the app with ngrok tunnel
if __name__ == '__main__':
    # Open a tunnel on the default Flask port 5000
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")

    # Run Flask app
    app.run()
