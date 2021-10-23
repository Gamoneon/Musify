# importing libraries
import joblib
import numpy as np
import librosa.feature
from keras.models import load_model

# song_name = r"C:\Users\anand\PycharmProjects\Musify\res\genres\metal\metal.00027.wav"
song_name = r"C:\Users\anand\Downloads\Miles Davis - So What - 2013_jazz_wav.wav"
genre = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
features = []

y, sr = librosa.load(song_name, mono=True)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rms = librosa.feature.rms(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y=y)
# tempo = librosa.feature.tempogram(y=y,sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

def addFeature(feature):
    features.append(np.mean(feature))
    # features.append(np.var(feature))

addFeature(chroma_stft)
addFeature(rms)
addFeature(spec_cent)
addFeature(spec_bw)
addFeature(rolloff)
addFeature(zcr)
for e in mfcc:
    addFeature(e)

print(features)
genre_scaler = joblib.load("genre_scaler.pkl")
X = genre_scaler.transform(np.array(features, dtype="float").reshape(1,-1))

model = load_model("genre.h5")
# print(model.summary())
predictions = model.predict(X)
print(predictions[0])
print(genre[np.argmax(predictions[0])])