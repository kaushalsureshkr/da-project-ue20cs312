from flask import Flask, render_template, request
import keras
import tensorflow as tf 
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import joblib


app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = "data"

def model_eeg_predict(inputs):
	# model_eeg = pickle.load(open('Model/eeg.pkl','rb'))
	model_eeg = tf.keras.models.load_model('Model/eeg')
	results_eeg=model_eeg.predict(inputs)
	#return results_eeg
	if (results_eeg[0][0]<0.5):
		return "There is a high chance of the applicant to have dyslexia."
	else:
		return "There is a low chance of the applicant to have dyslexia."

def model_stdtest_predict(inputs):
	model_stdtest = pickle.load(open('Model/stdtest.pkl','rb'))
	results_stdtest=model_stdtest.predict(inputs)
	if(results_stdtest == 0):
		return "There is a high chance of the applicant to have dyslexia."
	elif(results_stdtest == 1):
		return "There is a moderate chance of the applicant to have dyslexia."
	else:
		return "There is a low chance of the applicant to have dyslexia."
	#return results_stdtest'

def model_handwriting_predict(filepath):
	model_handwriting = tf.keras.models.load_model('Model/handwriting')
	img = tf.keras.utils.load_img(filepath, target_size=(29, 29))
	x = tf.keras.utils.img_to_array(img)
	x /= 255
	x = np.expand_dims(x, axis=0)
	result = model_handwriting.predict(x)
	#print(result)
	max_value = max(list(result[0]))
	#print(max_value)
	results = list(result[0])
	class_value = results.index(max_value)
	#print(class_value)

	if class_value == 0:
		return "The Person has written a CORRECTED letter, hence the chances of him having dyslexia is LOW."
	elif class_value == 1:
		return "The Person has written a NORMAL letter, hence the chances of him having dyslexia cannot be predicted with only this letter."
	elif class_value ==2:
		return "The Person has written a REVERSAL letter, hence the chances of him having dyslexia is HIGH."
	else:
		return "Could not predict with this letter, try using more letters to predict."

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/eeg', methods =["GET", "POST"])
def openeeg():
    return render_template("eeg.html")

@app.route('/stdtest', methods =["GET", "POST"])
def openstdtest():
    return render_template("std_tests.html")

@app.route('/handwriting', methods =["GET", "POST"])
def openhandwriting():
    return render_template("handwriting.html")

@app.route('/final_stdtest', methods =["GET", "POST"])
def sendstdtest():
	if request.method ==  "POST":
		lang = request.form.get("lang")
		memory = request.form.get("memory")
		speed = request.form.get("speed")
		visual = request.form.get("visual")
		audio = request.form.get("audio")
		survey = request.form.get("survey")
		arr_stdtest= np.array([[lang,memory,speed,visual,audio,survey]])
		result_stdtest=model_stdtest_predict(arr_stdtest)
	return render_template("result_stdtest.html",lang=lang,memory=memory,speed=speed,visual=visual,audio=audio,survey=survey,result_stdtest=result_stdtest)

@app.route('/final_eeg', methods =["GET", "POST"])
def sendeeg():
	if request.method ==  "POST":
		subid = float(request.form.get("subid"))
		age = float(request.form.get("age"))
		vidid = float(request.form.get("vidid"))
		attent = float(request.form.get("attent"))
		mediat = float(request.form.get("mediat"))
		raw = float(request.form.get("raw"))
		delta = float(request.form.get("delta"))
		theta = float(request.form.get("theta"))
		alphaa = float(request.form.get("alphaa"))
		alphab = float(request.form.get("alphab"))
		betaa = float(request.form.get("betaa"))
		betab = float(request.form.get("betab"))
		gammaa = float(request.form.get("gammaa"))
		gammab = float(request.form.get("gammab"))
		arr_eeg= np.array([[subid,vidid,attent,mediat,raw,delta,theta,alphaa,alphab,betaa,betab,gammaa,gammab,age]])
		sc=joblib.load('scaler_eeg.bin')
		df_sc=sc.transform(arr_eeg)
		result_eeg=model_eeg_predict(df_sc)
	return render_template("result_eeg.html",subid=subid,age=age,vidid=vidid,attent=attent,mediat=mediat,raw=raw,delta=delta,theta=theta,alphaa=alphaa,alphab=alphab,betaa=betaa,betab=betab,gammaa=gammaa,gammab=gammab,result_eeg=result_eeg)

@app.route('/final_handwriting',  methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		print(secure_filename("./data/" + f.filename))
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
		f.save(filepath)
		result_handwriting = model_handwriting_predict(filepath)
		return render_template('result_handwriting.html', result_handwriting = result_handwriting)	

if __name__=='__main__':
    app.run(debug=True)
	




