"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA DAS
"""

"""Create pitch profile template for 12 major and 12 minor chords and save them in a json file
Gmajor template = [1,0,0,0,1,0,0,1,0,0,0,0] - needs to be run just once"""

import json

template = dict()
major = ["G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#"]
minor = ["Gm", "G#m", "Am", "A#m", "Bm", "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m"]
offset = 0
num_chords = len(major)

# initialise lists with zeros
for chord in range(num_chords):
    template[major[chord]] = list()
    template[minor[chord]] = list()
    for note in range(num_chords):
        template[major[chord]].append(0)
        template[minor[chord]].append(0)

for chord in range(num_chords):
    for note in range(num_chords):
        if note == 0 or note == 7:
            template[major[chord]][(note + offset) % num_chords] = 1
            template[minor[chord]][(note + offset) % num_chords] = 1
        elif note == 4:
            template[major[chord]][(note + offset) % num_chords] = 1
        elif note == 3:
            template[minor[chord]][(note + offset) % num_chords] = 1
    offset += 1

# debugging
for key, value in template.items():
    print(key, value)

# save as JSON file
with open("chord_templates.json", "w") as fp:
    json.dump(template, fp, sort_keys=False)
    print("Saved succesfully to JSON file")
