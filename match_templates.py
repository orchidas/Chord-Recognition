"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA
"""

"""Correlate chord with existing binary chord templates to find best batch"""
import numpy as np 
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import json
from chromagram import compute_chroma

"""read from JSON file to get chord templates"""
with open('chord_templates.json', 'r') as fp:
    templates_json = json.load(fp)

chords = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
templates = []

for chord in chords:
	templates.append(templates_json[chord])
#print templates[1]

"""read audio and compute chromagram"""
directory = os.getcwd() + '/test_chords/';
fname = 'Grand Piano - Fazioli - minor chords - Am higher.wav';
(fs,s) = read(directory + fname)
chroma = compute_chroma(s,fs)


"""Correlate 12D chroma vector with each of 24 major and minor chords"""
cor_vec = np.zeros(24)
for n in range(24):
	cor_vec[n] = np.correlate(chroma, np.array(templates[n])) 

id_chord =  chords[np.argmax(cor_vec)]
print 'The identified chord is', id_chord 

plt.figure(4)
plt.stem(cor_vec)
plt.xticks(np.arange(24),chords)
plt.xlabel('Chord')
plt.ylabel('Correlation Coefficients')
plt.title('Identifiled chord is ' + id_chord)
plt.show()
