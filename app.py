from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

#@app.route('/preview')
#def preview():
#    df = pd.read_csv("data/iris.csv")
#    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
        if request.method == 'POST':
                age = request.form['age']
                sex = request.form['sex']
                cp = request.form['cp']
                oldpeak = request.form['oldpeak']
                ca = request.form['ca']
		#model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
                sample_data = [age,sex,cp,oldpeak,ca]
                clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
                ex1 = np.array(clean_data).reshape(1,-1)

		#ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		
                logit_model = joblib.load('data/logit_model_iris.pkl')
                result_prediction = logit_model.predict(ex1)

        return  render_template('index.html', sex=sex,
                age=age,
                cp=cp,
                ca=ca,
                oldpeak=oldpeak,                
                result_prediction=result_prediction)


if __name__ == '__main__':
	app.run(debug=True)
