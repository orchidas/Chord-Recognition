# Chord-Recognition
Automatic chord recognition in Python

Chords are identified automatically from monophonic/polyphonic audio. The feature extracted is called the <i>Pitch Class Profile</i>, which is obtained 
by computing the <i>Constant Q Transform</i>. Two methods are used for classification:
<ol>
<li>
Template matching - The pitch profile class is correlated with 24 major and minor chords, and the chord with highest correlation is identified.
Details given in the paper <i><a href = "https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.93.4283&rep=rep1&type=pdf">Automatic Chord Recognition from Audio Using Enhanced Pitch
  Class Profile</a></i> - Kyogu Lee in Proc. of ICMC, 2006. 
</li>
<li>
Hidden Markov Model - HMM is trained based on music theory according to the paper <i><A HREF = "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.375.2151&rep=rep1&type=pdf">A Robust Mid-level Representation for Harmonic Content in Music 
  Signals</a></i> - Juan P. Bello, Proc. of ISMIR, 2005. Viterbi decoding is used to estimate chord sequence in multi-timral, polyphonic music.
</ol>
