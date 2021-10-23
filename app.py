import csv
import os
import joblib
import numpy as np
import pandas as pd
import librosa.feature
from keras.models import load_model
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, flash, redirect
import subprocess

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(13)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

path = os.getcwd()

UPLOAD_FOLDER = os.path.join(path, "uploads")
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if not os.path.exists('uploads/data.csv'):
        new_file = open("uploads/data.csv", 'w')
        new_file.write("filename,genre\n")
        new_file.close()

    records = pd.read_csv('uploads/data.csv')
    rev_records = records.iloc[::-1]
    return render_template('index.html', records=rev_records.to_dict(orient="records"))


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Please select file for uploading')
            return redirect(request.url+"#uploadFile")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            song_file_name = path+"/uploads/"+filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if filename.rsplit('.', 1)[1].lower() == "mp3" and os.path.exists(song_file_name):
                AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
                sound = AudioSegment.from_mp3(song_file_name)
                name,ext = os.path.splitext(song_file_name)
                sound.export(path+"/uploads/"+name, format="wav")
                # subprocess.call(['ffmpeg', '-i', url_for('uploads', filename=name), "uploads/"+name])# convert mp3 to wav
                os.remove(song_file_name)
                song_file_name = path + "/uploads/" + name+".wav"

            #flash("File successfully uploaded!",song_file_name)
            genre = make_predictions(song_file_name)
            flash(genre)

            with open('uploads/data.csv', 'a', newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([file.filename, genre])

            file.close()
            return redirect(request.url+"/#uploadFile")
        else:
            flash('Allowed file types are'," ".join(ALLOWED_EXTENSIONS))
            return redirect(request.url)
    else:
        return redirect(request.url)


def make_predictions(file):
    model = load_model("CNN Model/genre.h5")
    genre_scaler = joblib.load("CNN Model/genre_scaler.pkl")
    genre = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    features = []

    y, sr = librosa.load(file, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features.append(np.mean(chroma_stft))
    features.append(np.mean(rms))
    features.append(np.mean(spec_cent))
    features.append(np.mean(spec_bw))
    features.append(np.mean(rolloff))
    features.append(np.mean(zcr))
    for e in mfcc:
        features.append(np.mean(e))

    X = genre_scaler.transform(np.array(features, dtype="float32").reshape(1,-1))
    predictions = model.predict(X)
    return genre[np.argmax(predictions[0])]


if __name__ == '__main__':
    app.run(debug=False)
