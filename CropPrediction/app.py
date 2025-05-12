from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)
model = joblib.load('dt.pkl')  # Load the trained model

# Label map for prediction
label_map = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

@app.route('/')
def home():
    prediction = request.args.get("prediction")
    return render_template('index.html', prediction_text=prediction, input_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {key: request.form[key] for key in ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']}
        
        # Convert inputs to floats for prediction
        features = [float(input_data[x]) for x in input_data]
        
        # Make prediction
        prediction_encoded = model.predict([features])[0]
        crop_name = label_map.get(prediction_encoded, "Unknown")
        
        # Pass the form data and prediction back to the template
        return render_template('index.html', 
                               prediction_text=f"Recommended Crop: {crop_name}",
                               nitrogen=input_data['nitrogen'], phosphorus=input_data['phosphorus'],
                               potassium=input_data['potassium'], temperature=input_data['temperature'],
                               humidity=input_data['humidity'], ph=input_data['ph'],
                               rainfall=input_data['rainfall'])
    except Exception as e:
        # Handle error and pass it back to the template
        return render_template('index.html', 
                               prediction_text=f"Error: {str(e)}",
                               nitrogen=request.form['nitrogen'], phosphorus=request.form['phosphorus'],
                               potassium=request.form['potassium'], temperature=request.form['temperature'],
                               humidity=request.form['humidity'], ph=request.form['ph'],
                               rainfall=request.form['rainfall'])

if __name__ == "__main__":
    app.run(debug=True)
