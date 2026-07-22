from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load encoders
le_gender = pickle.load(open("gender_encoder.pkl", "rb"))
le_city = pickle.load(open("city_encoder.pkl", "rb"))
le_membership = pickle.load(open("membership_encoder.pkl", "rb"))
le_satisfaction = pickle.load(open("satisfaction_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template(
        'index.html',
        genders=le_gender.classes_,
        cities=le_city.classes_,
        memberships=le_membership.classes_
    )

@app.route('/predict', methods=['POST'])
def predict():
    gender = le_gender.transform([request.form['gender']])[0]
    age = int(request.form['age'])
    city = le_city.transform([request.form['city']])[0]
    membership = le_membership.transform([request.form['membership']])[0]
    total_spend = float(request.form['total_spend'])
    items = int(request.form['items'])
    rating = float(request.form['rating'])
    discount = 1 if request.form['discount'] == 'Yes' else 0
    days = int(request.form['days'])

    features = np.array([[gender, age, city, membership, total_spend,
                          items, rating, discount, days]])

    prediction = model.predict(features)[0]
    result = le_satisfaction.inverse_transform([prediction])[0]

    return render_template(
        'index.html',
        prediction=result,
        genders=le_gender.classes_,
        cities=le_city.classes_,
        memberships=le_membership.classes_
    )

if __name__ == '__main__':
    app.run(debug=True)
