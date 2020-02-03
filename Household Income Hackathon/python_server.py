from flask import Flask, request
import pickle
import numpy as np
import json
from pandas import DataFrame

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def predict():
    data = request.get_json()

    male = data['gender']['male']

    no_hs = data['education']['no-hs']
    hs = data['education']['hs']
    bs = data['education']['bs']
    ms_dr = data['education']['ms-dr']

    white = data['race']['white']
    black = data['race']['black']
    asian = data['race']['asian']
    other = data['race']['other']

    married = data['marital']['married']
    single = data['marital']['single']
    other = data['marital']['other']

    age1 = data['age']['18-24']
    age2 = data['age']['25-34']
    age3 = data['age']['35-44']
    age4 = data['age']['45-54']
    age5 = data['age']['55-64']

    X = DataFrame()
    X['race_white'] = [white]
    X['race_black'] = [black]
    X['race_american-indian'] = [np.NaN]
    X['race_asian'] = [asian]
    X['race_native-hawaiian'] = [np.NaN]
    X['race_other'] = [other]
    X['race_multiracial'] = [np.NaN]
    X['Male'] = [male]
    X['18-24'] = [age1]
    X['25-34'] = [age2]
    X['35-44'] = [age3]
    X['45-54'] = [age4]
    X['55-64'] = [age5]
    X['No Schooling'] = [np.NaN]
    X['Primary School'] = [no_hs]
    X['Middle School'] = [np.NaN]
    X['High School (No Diploma)'] = [np.NaN]
    X['High School Diploma or Equivalent'] = [hs]
    X['College (No Degree)'] = [np.NaN]
    X['Associate Degree'] = [np.NaN]
    X['Bachelors Degree'] = [bs]
    X['Masters Degree'] = [ms_dr/2.]
    X['Professional School Degree'] = [np.NaN]
    X['Doctorate Degree'] = [ms_dr/2.]
    X['Married Present'] = [married]
    X['Married Absent'] = [np.NaN]
    X['Never Married'] = [single]
    X['Divorced'] = [np.NaN]
    X['Widowed'] = [np.NaN]
    X['hispanic_percentage'] = [np.NaN]

    y = model.predict(X)

    return json.dumps(int(y[0]))


if __name__ == '__main__':

    with open('xgboost.model', 'rb') as f:
        model = pickle.load(f)

    app.run(debug=True)
