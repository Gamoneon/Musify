import librosa
import pandas as pd
import numpy as np
import os
import csv
import joblib
import random
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
import keras
from keras import models
from keras import layers

# generating a dataset
# header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()
#
# with open('data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# for g in genres:
#     print("\nAnalyzing",g,"Genre")
#     for filename in os.listdir(f'C:/Users/anand/PycharmProjects/Musify/res/genres/{g}'):
#         print(filename)
#         songname = f'C:/Users/anand/PycharmProjects/Musify/res/genres/{g}/{filename}'
#
#         # y is a 1Dimensional array of time series and sr is a sampling rate (Default SR is 22KHz)
#         y, sr = librosa.load(songname, mono=True, duration=30)
#
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         rms = librosa.feature.rms(y=y)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)        # [[1st],[2nd],[3rd],[4th]....[20th]] 20 lists in a list
#
#         to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {g}'
#
#         with open('data.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())

# reading dataset from csv
data = pd.read_csv('data.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'], axis=1)
# print(data.head())

# Encoding lables
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# Normalizing dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
joblib.dump(scaler, 'genre_scaler.pkl')


# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128)

# calculate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy: ', test_acc*100,"\nTest Loss",test_loss)
model.save("genre.h5")

# predictions
predictions = model.predict(X_test)
for i in range(10):
    x = random.randint(0,200)
    print(genres[np.argmax(predictions[x])])