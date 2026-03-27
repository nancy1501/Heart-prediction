import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

df = pd.read_csv("heart.csv")

ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean).round(2)

resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean).round(2)

df_encode = pd.get_dummies(df, drop_first=True)

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()

X = df_encode.drop('HeartDisease', axis=1)
y = df_encode['HeartDisease']

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

feature_cols = X.columns.tolist()


def preprocess_input(data):
    age = float(data['age'])
    sex = data['sex']
    chest_pain = data['chest_pain_type']
    resting_bp = float(data['resting_bp'])
    cholesterol = float(data['cholesterol'])
    fasting_bs = int(data['fasting_bs'])
    resting_ecg = data['resting_ecg']
    max_hr = float(data['max_hr'])
    exercise_angina = data['exercise_angina']
    oldpeak = float(data['oldpeak'])
    st_slope = data['st_slope']

    if cholesterol == 0:
        cholesterol = ch_mean
    if resting_bp == 0:
        resting_bp = resting_bp_mean

    row = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': True if sex == 'M' else False,
        'ChestPainType_ATA': True if chest_pain == 'ATA' else False,
        'ChestPainType_NAP': True if chest_pain == 'NAP' else False,
        'ChestPainType_TA': True if chest_pain == 'TA' else False,
        'RestingECG_Normal': True if resting_ecg == 'Normal' else False,
        'RestingECG_ST': True if resting_ecg == 'ST' else False,
        'ExerciseAngina_Y': True if exercise_angina == 'Y' else False,
        'ST_Slope_Flat': True if st_slope == 'Flat' else False,
        'ST_Slope_Up': True if st_slope == 'Up' else False,
    }

    input_df = pd.DataFrame([row])
    input_df = input_df[feature_cols]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    return input_df


@app.route('/')
def index():
    return render_template('index.html', accuracy=round(accuracy * 100, 2))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = preprocess_input(data)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        result = {
            'prediction': int(prediction),
            'probability_no_disease': round(float(probability[0]) * 100, 1),
            'probability_disease': round(float(probability[1]) * 100, 1),
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
