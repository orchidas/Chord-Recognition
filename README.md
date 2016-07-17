# Chord-Recognition
Automatic chord recognition in Python

Chords are identified automatically from monophonic/polyphonic audio. The feature extracted is called the <i>Pitch Class Profile</i>, which is obtained 
by computing the <i>Constant Q Transform</i>. Two methods are used for classification:
<ol>
<li>
Template matching - The pitch profile class is correlated with 24 major and minor chords, and the chord with highest correlation is identified.
Details given in the paper <b>Automatic Chord Recognition from Audio Using Enhanced Pitch
Class Profile </b> - <i>Kyogu Lee, CCRMA Stanford</i>. Works well
for monophonic music.
</li>
<li>
Hidden Markov Model - HMM is trained based on music theory according to the paper <b>A Robust Mid-level Representation for Harmonic Content in Music 
Signals</b> - <i>Juan P. Bello, MARL NYU</i>. Viterbi decoding is used to estimate chord sequence in polyphonic music.
</ol>
