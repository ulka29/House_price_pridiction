import pandas as pd
from flask import Flask,render_template, request
import pickle


data = pd.read_csv("cleaned_data.csv")
app = Flask(__name__)
pipe= pickle.load(open('RidgeModel1.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=float(request.form.get('bhk'))
    bath=float(request.form.get('bathrooms'))
    sqft=float(request.form.get('sqft'))

    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft', 'bath', 'bhk'])
    print(input.info())

    pridiction = pipe.predict(input)[0]
    return str(pridiction)

if __name__=="__main__":


    app.run()
