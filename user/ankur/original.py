from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mido

midi = MidiFile('Suteki_Da_Ne_(Piano_Version).mid')

# preprocess the midi file
notes = []

time = float(0)
prev = float(0)
# The tracks attribute is a list of tracks.
# Each track is a list of messages and meta messages, with the time attribute of each messages set to its delta time (in ticks)
for msg in midi:
    time += msg.time
    if not msg.is_meta:  # easy to playback on port
        # only interested in piano channel
        if msg.channel == 0:
            if msg.type == 'note_on':
                note = msg.bytes()
                # only interested in the note and velocity. note message is in the form [type, note, velocity]
                note = note[1:3]
                note.append(time - prev)
                prev = time
                notes.append(note)

# need to scale notes
n = []
for note in notes:
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
print(x[1])
print(y[1])
x = np.array(x)  # convert to numpy arrays to pass it through model
y = np.array(y)
print(x[1])
print(y[1])

model = Sequential()
model.add(LSTM(512, input_shape=(20, 3), return_sequences=True))  # np is 20*3
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))  # return_sequences=False #return_sequences=False
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(3, activation="softmax"))  # output=3

model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
model.fit(x, y, epochs=1000, batch_size=200, validation_split=0.1)

seed = notes[0:n_p]
x = seed
x = np.expand_dims(x, axis=0)
print(x)
predict = []
for i in range(2000):
    p = model.predict(x)
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

from mido import MidiFile, MidiTrack, Message

# saving track from bytes data
m = MidiFile()
track = MidiTrack()
m.tracks.append(track)

for note in predict:
    # 147 means note_on
    note = np.insert(note, 0, 147)
    bytes = note.astype(int)
    print(note)
    msg = Message.from_bytes(bytes[0:3])
    time = int(note[3] / 0.001025)  # to rescale to midi's delta ticks. arbitrary value
    msg.time = time
    track.append(msg)

m.save('Ai_song.mid')
