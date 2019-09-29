import numpy as np
import os

from mido import MidiFile, MidiTrack, Message

# from sklearn.externals import joblib
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam,SGD,Adagrad

folder_name = '/home/ubuntu/hackathon/ai_composer/midi'
midis = []
for filename in os.listdir(folder_name):
    midis.append(MidiFile(os.path.join(folder_name, filename)))
# midi = MidiFile('/home/ubuntu/Downloads/Super Mario Bros. 1 - Super Mario Bros - Main Theme with Left Hand Chords.mid')
# midi = MidiFile('Ai_song.mid')
# print(midi)


# def save_model(model, filename):
#     joblib.dump(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))

def preprocess():
    notes = []

    for midi in midis:
        time = float(0)
        prev = float(0)
        for track in midi.tracks:
            for msg in track:
                time += msg.time
                # print("time=", time)
                if not msg.is_meta:  # easy to playback on port
                    # only interested in piano channel
                    # print(msg.bytes())
                    # print(msg.channel)
                    # print(msg.type)
                    try:
                        if msg.channel == 0:
                            if msg.type == 'note_on':
                                note = msg.bytes()
                                # [only interested in the note and velocity. note message is in the form [type, note, velocity]]
                                note = note[1:3]
                                note.append(time - prev)
                                prev = time
                                notes.append(note)
                    except:
                        pass

    # need to scale notes
    n = []
    for note in notes:
        # print(note)
        note[0] = (note[0] - 24) / 88
        note[1] = note[1] / 127
        n.append(note[2])
    max_n = max(n)  # scale based on the biggest time of any note
    for note in notes:
        note[2] = note[2] / max_n

    x = []
    y = []
    n_p = 20
    for i in range(len(notes) - n_p):
        current = notes[i:i + n_p]
        next = notes[i + n_p]
        x.append(current)
        y.append(next)
    x = np.array(x)  # convert to numpy arrays to pass it through model
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    print(x[1])
    print(y[1])


    model = Sequential()
    model.add(LSTM(512, input_shape=(20, 3), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))  # return_sequences=False
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax"))  # output=3

    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
    model.fit(x, y, epochs=10, batch_size=200, validation_split=0.1)


    seed = notes[0:n_p]
    x = seed
    x = np.expand_dims(x, axis=0)
    print(x)
    predict = []
    for i in range(200):
        p = model.predict(x)
        print(p)
        # print()
        # print(p)
        x = np.squeeze(x)  # squeezed to concateneate
        x = np.concatenate((x, p))
        x = x[1:]
        x = np.expand_dims(x, axis=0)  # expanded to roll back
        p = np.squeeze(p)
        predict.append(p)

    # unrolling back from conversion
    for a in predict:
        a[0] = int(88 * a[0] + 24)
        a[1] = int(127 * a[1])
        a[2] *= max_n
        # a[2] /= 1000
        # reject values out of range  (note[0]=24-102)(note[1]=0-127)(note[2]=0-__)
        if a[0] < 24:
            a[0] = 24
        elif a[0] > 102:
            a[0] = 102
        if a[1] < 0:
            a[1] = 0
        elif a[1] > 127:
            a[1] = 127
        if a[2] < 0:
            a[2] = 0
        print(a)
        print()
    # print(predict)

    # saving track from bytes data
    m = MidiFile()
    track = MidiTrack()
    m.tracks.append(track)

    print("---------------------------------------")
    for note in predict:
        # 147 means note_on
        print(note)
        note = np.insert(note, 0, 147)
        print(note)
        print("-------------------------------")
        bytes = note.astype(int)
        # print(note)
        msg = Message.from_bytes(bytes[0:3])
        print(note)
        time = int(note[3]/ 5)  # to rescale to midi's delta ticks. arbitrary value
        msg.time = time
        track.append(msg)

    m.save('Ai_song.mid')


preprocess()
# create_model(x, y)
