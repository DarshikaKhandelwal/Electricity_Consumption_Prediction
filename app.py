from flask import Flask, render_template, request
import xgboost as xgb
import pickle
import pandas as pd

model = pickle.load(open('xgb_model.pkl', 'rb'))
model = xgb.Booster()
model.load_model('model.txt')


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods = ['GET', 'POST'])
def prediction():
    if request.method == "POST":
       # Convert date string to datetime
        date = request.form['date']
        temperature = request.form['temperature']
        dt = pd.to_datetime(date, format='%b-%y')
        temp = float(temperature)
        month = dt.month
        year = dt.year
        data = {'month': [month], 'year': [year], 'temperature':[temp]}
        df = pd.DataFrame(data)
        dtest = xgb.DMatrix(df, enable_categorical=True)
        prediction = model.predict(dtest)
        return render_template("prediction.html", prediction_text = "The electricity consumption prediction is {} MW".format(prediction))
    
    else:
        return render_template("prediction.html")
@app.route("/about", methods = ['GET', 'POST'])
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug = False, host = '0.0.0.0')


