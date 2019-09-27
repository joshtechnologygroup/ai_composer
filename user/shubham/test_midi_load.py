import numpy as np

from mido import MidiFile

midi = MidiFile('/home/ubuntu/Downloads/Super Mario Bros. 1 - Super Mario Bros - Main Theme with Left Hand Chords.mid')
# print(midi)


def preprocess():
    notes = []

    time = float(0)
    prev = float(0)
    for track in midi.tracks:
        for msg in track:
            time += msg.time
            print("time=", time)
            if not msg.is_meta:  # easy to playback on port
                # only interested in piano channel
                if msg.channel == 0:
                    if msg.type == 'note_on':
                        note = msg.bytes()
                        # [only interested in the note and velocity. note message is in the form [type, note, velocity]]
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
    x = np.array(x)  # convert to numpy arrays to pass it through model
    y = np.array(y)
    print(x[1])
    print(y[1])

preprocess()
