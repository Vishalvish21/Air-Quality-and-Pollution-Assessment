from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("final_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        features = [float(request.form[key]) for key in request.form.keys()]
        features_array = np.array([features])  # Convert to NumPy array

        # Make prediction
        prediction = model.predict(features_array)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
