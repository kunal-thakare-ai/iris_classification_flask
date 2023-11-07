import pickle
import numpy as np
from flask import Flask, render_template, request
# import flasgger
# from flasgger import Swagger


app = Flask(__name__)
# Swagger(app)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare the input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make predictions
        prediction = model.predict(input_data)

        return render_template('prediction.html', prediction=prediction[0])
    return render_template('index.html')

@app.route("/predict")
def pred():
    return "1"
if __name__ == '__main__':
    app.run(debug=False)
