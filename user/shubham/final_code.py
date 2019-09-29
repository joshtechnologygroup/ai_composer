import numpy as np
import os
import random

from mido import MidiFile, MidiTrack, Message

# from sklearn.externals import joblib
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam,SGD,Adagrad
from sklearn.externals import joblib

folder_name = '/home/ubuntu/Downloads/new_midis'
test = '/home/ubuntu/Downloads/new_midis/Suteki_Da_Ne_(Piano_Version).mid'
n_p = 20
MODEL_FILE_PATH = 'user/shubham/models/trained_model.sav'
MELODY_FILE_PATH = 'Ai_song.mid'

def sample(preds, temperature=1.0):
    if np.count_nonzero(preds) == 0:
        return 0
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    loaded_model = joblib.load(filename)
    return loaded_model

def fetch_data(filepath):
    midi_file_obj = MidiFile(filepath)
    notes = []
    hash = {}
    for track in midi_file_obj.tracks:
        for msg in track:
            if not msg.is_meta:
                try:
                    if msg.channel == 0:
                        if msg.type == 'note_on':
                            if msg.velocity != 0:
                                check = [0] * 456
                                note = msg.bytes()
                                check[note[1]] = 1
                                check[note[2] + 128] = 1
                                check[int(msg.time % 1000 / 10) + 256] = 1
                                notes.append(check)
                                hash[msg.note] = note
                            else:
                                hash[msg.note].append(int(msg.time % 1000 / 10) + 356)
                                del hash[msg.note]
                except:
                    pass
    return notes

def preprocess(folder_name):
    music_notes = []
    for filename in os.listdir(folder_name):
        music_notes.append(fetch_data(os.path.join(folder_name, filename)))

    X = []
    Y = []
    for notes in music_notes:
        x = []
        y = []
        for i in range(len(notes) - n_p):
            current = notes[i:i + n_p]
            next = notes[i + n_p]
            x.append(current)
            y.append(next)

        X += x
        Y += y

    return np.array(X), np.array(Y)

def train_model(epochs=10, batch_size=50):
    input_data, output_data = preprocess(folder_name)

    model = Sequential()
    model.add(LSTM(512, input_shape=(n_p, 456), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(456, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
    model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    save_model(model, MODEL_FILE_PATH)

def predict(number_of_predcitions=100):
    notes = fetch_data(test)
    model = load_model(MODEL_FILE_PATH)

    seed = notes[0:n_p]
    x = seed
    x = np.expand_dims(x, axis=0)

    predict = []
    for i in range(number_of_predcitions):
        p = model.predict(x)

        # print('--------------- calculating flags')
        note = sample(p[0][:128], 0.75)
        # print('-------------- calculating velocity')
        velocity = sample(p[0][128:256], 0.75)
        # print('-------------- calculating time1')
        time1 = sample(p[0][256:356], 0.75)
        # print('-------------- calculating time2')
        time2 = sample(p[0][356:], 0.75)
        # print('--------------- flags calculated')

        a = [0] * 456
        a[note] = 1
        a[velocity + 128] = 1
        a[time1 + 256] = 1
        a[time2 + 356] = 1

        x = np.squeeze(x)  # squeezed to concateneate
        a = [a]
        x = np.concatenate((x, a))
        x = x[1:]
        x = np.expand_dims(x, axis=0)  # expanded to roll back
        p = np.squeeze(a)
        predict.append(p)

    normalized = []
    for a in predict:
        check = []
        count = 0
        for i, x in enumerate(a):
            if x == 1:
                check.append(i)
                count += 1

        normalized.append(check)

    m = MidiFile()
    track = MidiTrack()
    m.tracks.append(track)

    for note in normalized:
        note[1] = note[1] - 128
        note[2] = (note[2] - 256) * 10
        note[3] = (note[3] - 356) * 10

        l1 = [147, note[0], note[1], note[2]]
        l2 = [147, note[0], 0, note[3]]
        note1 = np.asarray(l1)
        note2 = np.asarray(l2)
        print(note1)
        print(note2)
        bytes1 = note1.astype(int)
        bytes2 = note2.astype(int)

        msg1 = Message.from_bytes(bytes1[0:3])
        msg2 = Message.from_bytes(bytes2[0:3])

        time1 = int(note1[3])
        time2 = int(note2[3])

        msg1.time = time1
        msg2.time = time2

        track.append(msg1)
        track.append(msg2)

    m.save(MELODY_FILE_PATH)

train_model()
predict()
