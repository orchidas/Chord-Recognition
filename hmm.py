"""Automatic chord recogniton with HMM, as suggested by Juan P. Bello in
'A mid level representation for harmonic content in music signals'
@author ORCHISAMA"""

from __future__ import division
from chromagram import compute_chroma
import os
from scipy.io.wavfile import read
import numpy as np 
import json


def assign_vals(indices):
	x = np.zeros(12)
	x[indices[0]] = 0.8
	x[indices[1]] = 0.6
	x[indices[2]] = 0.8
	return x

"""calculates multivariate gaussian matrix from mean and covariance matrices"""
def multivariate_gaussian(x, meu, cov):
	#normalize chroma vector
	x = x/np.max(x)
	det = np.linalg.det(cov)
	val = -0.5 * np.dot(np.dot((x-meu).T, np.linalg.inv(cov)), (x-meu))
	val /= np.sqrt(((2*np.pi)**12)*det)	
	return val


"""initialize the emission, transition and initialisation matrices for HMM in chord recognition
PI - initialisation matrix, #A - transition matrix, #B - observation matrix"""
def initialize(chroma, templates, nested_cof):

	"""initialising PI with equal probabilities"""
	PI = np.ones(24)/24

	"""initialising A based on nested circle of fifths"""
	eps = 0.1
	A = np.empty((24,24))
	for chord in chords:
		ind = nested_cof.index(chord)
		t = ind
		for i in range(24):
			if t >= 24:
				t = t%24
			A[ind][t] = (abs(12-i)+eps)/(144 + 24*eps)
			t += 1


	"""initialising based on tonic triads - Mean matrix; Tonic with dominant - 0.8,
	tonic with mediant 0.6 and mediant-dominant 0.8, non-triad diagonal	elements 
	with 0.2 - covariance matrix"""

	nFrames = np.shape(chroma)[1]
	B = np.zeros((24,nFrames))
	meu_mat = np.zeros((24,12))
	cov_mat = np.zeros((24,12,12))
	meu_mat = np.array(templates)

	for i in range(24):
		args = np.where(meu_mat[i,:] == 1)
		#filling tonic triad
		for arg in np.nditer(args):
			ind = np.where(meu_mat[arg,:] == 1)
			cov_mat[i,arg,:] = assign_vals(ind[0])
		#filling diagonal elements
		for j in range(12):
			if cov_mat[i,j,j] == 0:
				cov_mat[i,j,j] = 0.2
			

	"""observation matrix B is a multivariate Gaussian calculated from mean vector and 
	covariance matrix"""
	for n in range(24):
		for m in range(nFrames):
			B[n,m] = multivariate_gaussian(chroma[:,m], meu_mat[n,:],cov_mat[n,:,:])

	return (PI,A,B)
	


"""Viterbi algorithm to find Path with highest probability - dynamic programming"""
def viterbi(PI,A,B):
	(nrow, ncol) = np.shape(B)
	path = np.zeros((nrow, ncol))
	states = np.zeros((nrow,ncol))
	path[:,0] = PI * B[:,0]

	for i in range(1,ncol):
		for j in range(nrow):
			s = [(path[k,i-1] * A[k,j] * B[j,i], k) for k in range(nrow)]
			(prob,state) = max(s)
			path[j,i] = prob
			states[j,i-1] = state
	
	return (path,states)



"""read from JSON file to get chord templates"""
with open('chord_templates.json', 'r') as fp:
	templates_json = json.load(fp)

chords = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
nested_cof = ['G','Bm','D','F#m','A','C#m','E','G#m','B','D#m','F#','A#m','C#',"Fm","G#",'Cm','D#','Gm','A#','Dm','F','Am','C','Em']
templates = []

for chord in chords:
	templates.append(templates_json[chord])

"""read audio and compute chromagram"""
directory = os.getcwd() + '/test_chords/';
fname = 'Grand Piano - Fazioli - major A middle.wav';
(fs,s) = read(directory + fname)

x = s[::4]
x = x[:,1]
fs = int(fs/4)

#framing audio 
nfft = 8192
hop_size = 1024
nFrames = int(np.round(len(x)/(nfft-hop_size)))
#zero padding to make signal length long enough to have nFrames
x = np.append(x, np.zeros(nfft))
xFrame = np.empty((nfft, nFrames))
start = 0   
chroma = np.empty((12,nFrames)) 
timestamp = np.zeros(nFrames)
#compute PCP
for n in range(nFrames):
	xFrame[:,n] = x[start:start+nfft] 
	start = start + nfft - hop_size 
	chroma[:,n] = compute_chroma(xFrame[:,n],fs)
	timestamp[n] = n*(nfft-hop_size)/fs 


#get max probability path from Viterbi algorithm
(PI,A,B) = initialize(chroma, templates, nested_cof)
(path, states) = viterbi(PI,A,B)

#normalize path
for i in range(nFrames):
	path[:,i] /= sum(path[:,i])

#choose most likely chord - with max value in 'path'
final_chords = []
indices = np.argmax(path,axis=0)
final_states = np.zeros(nFrames)
for i in range(nFrames):
	final_states[i] = states[indices[i],i]
	final_chords.append(chords[int(final_states[i])])

print 'Time(s)','Chords'
for i in range(nFrames):
	print timestamp[i], final_chords[i]



