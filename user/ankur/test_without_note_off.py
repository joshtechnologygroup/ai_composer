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

# midis = midis[10:18]

test = midis.pop()
# midi = MidiFile('/home/ubuntu/Downloads/Super Mario Bros. 1 - Super Mario Bros - Main Theme with Left Hand Chords.mid')
# midi = MidiFile('Ai_song.mid')
# print(midi)


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)


# def save_model(model, filename):
#     joblib.dump(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))

# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     return preds

def preprocess():
    notes = []

    test_notes = []
    for track in test.tracks:
        for msg in track:
            if not msg.is_meta:
                if msg.channel == 0:
                    if msg.type == 'note_on':
                        note = msg.bytes()
                        note.append(msg.time)
                        test_notes.append(note)

    n1 = []
    n2 = []
    for note in test_notes:
        # print(note)
        note[0] = (note[0] - 24) / 88
        note[1] = note[1] / 127
        n1.append(note[2])
        if len(note) == 3:
            note.append(150)
        n2.append(note[3])
    max_n = max(n1)  # scale based on the biggest time of any note
    max_n2 = max(n2)
    for note in test_notes:
        note[2] = note[2] / max_n
        note[3] = note[3] / max_n2


    for midi in midis:
        time = float(0)
        prev = float(0)

        hash = {}
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
                                if msg.velocity != 0:
                                    note = msg.bytes()
                                    # # [only interested in the note and velocity. note message is in the form [type, note, velocity]]
                                    note = note[1:3]
                                    note.append(msg.time)
                                    notes.append(note)
                                    hash[msg.note] = note
                                    # note.append(time - prev)
                                    # prev = time
                                    # notes.append(note)
                                else:
                                    hash[msg.note].append(msg.time)
                                    del hash[msg.note]

                    except:
                        pass

    # need to scale notes
    n1 = []
    n2 = []
    for note in notes:
        # print(note)
        note[0] = (note[0] - 24) / 88
        note[1] = note[1] / 127
        n1.append(note[2])
        if len(note) == 3:
            note.append(150)
        n2.append(note[3])
    max_n = max(n1)  # scale based on the biggest time of any note
    max_n2 = max(n2)
    for note in notes:
        note[2] = note[2] / max_n
        note[3] = note[3] / max_n2

    x = []
    y = []
    n_p = 50
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
    model.add(LSTM(128, input_shape=(n_p, 4), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))  # return_sequences=False
    # model.add(Dropout(0.3))
    # model.add(LSTM(256))
    # model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation="softmax"))  # output=3

    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
    model.fit(x, y, epochs=10, batch_size=100, validation_split=0.1)


    seed = test_notes[0:n_p]
    x = seed
    x = np.expand_dims(x, axis=0)
    print('input music file...........')
    print(x)
    predict = []
    for i in range(20):
        print('------------------------------')
        print(x)
        p = model.predict(x)
        # print('------------ Model predict')
        # print(p)
        # print('----------- After sampling')
        # p = sample(p, 0.75)
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
        a[3] *= max_n2
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
        if a[3] < 0:
            a[3] = 0
        print(a)
        print()
    # print(predict)

    # saving track from bytes data
    m = MidiFile()
    track = MidiTrack()
    m.tracks.append(track)

    for note in predict:
        # 147 means note_on
        # note = np.insert(note, 0, 147)
        # bytes = note.astype(int)
        # # print(note)
        # msg = Message.from_bytes(bytes[0:3])
        # time = int(note[3]/ 5)  # to rescale to midi's delta ticks. arbitrary value
        # msg.time = time
        # track.append(msg)

        l1 = [147, note[0], note[1], note[2]]
        l2 = [147, note[0], 0, note[3]]
        note1 = np.asarray(l1)
        note2 = np.asarray(l2)
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

    m.save('Ai_song.mid')


preprocess()
# create_model(x, y)
