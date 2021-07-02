import os
import pickle

from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
	global dog_race, color_result, dog_race_pred
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__)) + "/uploads/" + filename)
			dog_race_pred = dog_race(image)
			redirect(url_for('upload_file', filename=filename))
		return '''
				<!doctype html>
				<title>Results</title>
				<h1>Image contains a - </h1>
				<h2>1st - ''' + dog_race_pred[0] + '''</h2>
				<h2>or</h2>
				<h2>2nd - ''' + dog_race_pred[1] + '''</h2>
				<h2>or</h2>
				<h2>3rt - ''' + dog_race_pred[2] + '''</h2>
				<form method=post enctype=multipart/form-data>
				  <input type=file name=file>
				  <input type=submit value=Upload>
				</form>
				'''
	return '''
		<!doctype html>
		<title>Upload new File</title>
		<h1>Upload new File</h1>
		<form method=post enctype=multipart/form-data>
		  <input type=file name=file>
		  <input type=submit value=Upload>
		</form>
		'''


def dog_race(image):
	'''Determines if the image contains a cat or dog'''
	classifier = load_model('./models/model120.h5',	compile=False)
	races_invert_dict_file = open("./models/races_invert_dict.pkl", "rb")
	races_invert_dict = pickle.load(races_invert_dict_file)
	image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
	image = image.reshape(1, 224, 224, 3)
	prediction = classifier.predict(image)[0]
	top3 = prediction.argsort()[::-1][:3]
	top3_list = []
	for i in top3:
		print(races_invert_dict[i].split('-')[1], prediction[i])
		top3_list.append(races_invert_dict[i].split('-')[1] + " = " + str(prediction[i]))

	res = races_invert_dict[np.argmax(prediction)].split('-')[1]
	K.clear_session()
	return top3_list

if __name__ == "__main__":
	app.run()  # host= '0.0.0.0', port=80)
