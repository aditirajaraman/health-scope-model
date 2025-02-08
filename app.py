from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import flask_cors

import random

app = Flask(__name__)
flask_cors.CORS(app)

@app.route("/")
def hello_world():
    random_number = random.randint(1, 10)
    return f"<p>Hello, World! Random number: {random_number}</p>"


from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import flask_cors
import joblib
import random

app = Flask(__name__)
flask_cors.CORS(app)

df = pd.read_csv('water_potability.csv')
df.dropna(inplace=True)

X = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate']]
y = df['Potability']

print("Class distribution:\n", y.value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

X_min = X.min()
X_max = X.max()


@app.route("/")
def hello_world():
    random_number = random.randint(1, 10)
    return f"<p>Hello, World! Random number: {random_number}</p>"


@app.route("/predict", methods=["POST"])
def predict():
    inputs = request.json.get("inputs")
    
    if not inputs:
        return {"error": "No input data provided"}, 400

    try:
        ph = float(inputs.get("ph", 0))
        hardness = float(inputs.get("Hardness", 0))
        solids = float(inputs.get("Solids", 0))
        chloramines = float(inputs.get("Chloramines", 0))
        sulfate = float(inputs.get("Sulfate", 0))
    except ValueError:
        return {"error": "Invalid input values"}, 400

    print(f"Received inputs: {inputs}")

    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    column_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate']
    data = [[ph, hardness, solids, chloramines, sulfate]]
    df_input = pd.DataFrame(data, columns=column_names)

    df_input = df_input.clip(lower=X_min, upper=X_max, axis=1)

    df_input_scaled = scaler.transform(df_input)

    prediction = knn.predict(df_input_scaled)[0]

    print(f"Processed input after scaling: {df_input_scaled}")
    print(f"Prediction: {prediction}")

    return {"prediction": str(prediction)}


@app.route("/random_number", methods=["POST"])
def random_number_with_limit():
    upper_limit = request.json.get("upper_limit", 10)
    
    try:
        upper_limit = int(upper_limit)
        if upper_limit < 1:
            return {"error": "Upper limit must be greater than 0"}, 400
    except ValueError:
        return {"error": "Invalid upper limit"}, 400

    random_number = random.randint(1, upper_limit)
    return {"random_number": random_number}


if __name__ == "__main__":
    app.run(debug=True)
