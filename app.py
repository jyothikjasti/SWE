from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [[int(x) for x in request.form.values()]]
    final = np.array(int_features)
    print(final)

    col = np.array(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    df = pd.DataFrame(final, columns=col)
    prediction = model.predict(df)

    # print the output
    if prediction == 1:
        return render_template('index.html', pred='Loan can be approved')
    else:
        return render_template('index.html', pred='Loan cannot be approved')


if __name__ == '__main__':
    app.run(debug=True)
