from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder

label_encoders = {
    'Brand': LabelEncoder().fit(['Brand A', 'Brand B', 'Brand C']),  # Add actual brands
    'Type': LabelEncoder().fit(['Portable', 'Window', 'Split']),
    'Features': LabelEncoder().fit(['WiFi', 'Inverter Technology', 'Quiet', 'Standard']),
    'Location': LabelEncoder().fit(['USA', 'Europe', 'India', 'Australia'])
}

app = Flask(__name__)
model = pickle.load(open('ACM.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    brand = request.form.get('brand')
    capacity = float(request.form.get('capacity'))
    eer_seer = float(request.form.get('eer_seer'))  
    ac_type = request.form.get('type')
    features = request.form.get('features')
    location = request.form.get('location')
    age = int(request.form.get('age'))  

    try:
        
        brand_encoded = label_encoders['Brand'].transform([brand])[0]
    except ValueError:
        return jsonify({'error': 'Brand not recognized'})

    try:
        ac_type_encoded = label_encoders['Type'].transform([ac_type])[0]
    except ValueError:
        return jsonify({'error': 'AC Type not recognized'})

    try:
        features_encoded = label_encoders['Features'].transform([features])[0]
    except ValueError:
        return jsonify({'error': 'Feature not recognized'})

    try:
        location_encoded = label_encoders['Location'].transform([location])[0]
    except ValueError:
        return jsonify({'error': 'Location not recognized'})

   
    input_features = [[brand_encoded, capacity, eer_seer, ac_type_encoded, features_encoded, location_encoded, age]]

    
    predicted_price = model.predict(input_features)[0]
    print(f"Predicted Price: {predicted_price}")

   
    return jsonify({'price': float(predicted_price)}) 


if __name__ == '__main__':
    app.run(debug=True)
