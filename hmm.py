"""Automatic chord recogniton with HMM, as suggested by Juan P. Bello in
'A mid level representation for harmonic content in music signals'
@author ORCHISAMA DAS, 2016"""

from __future__ import division
from chromagram import compute_chroma
import os
import numpy as np


"""calculates multivariate gaussian matrix from mean and covariance matrices"""


def multivariate_gaussian(x, meu, cov):

    det = np.linalg.det(cov)
    val = np.exp(-0.5 * np.dot(np.dot((x - meu).T, np.linalg.inv(cov)), (x - meu)))
    try:
        val /= np.sqrt(((2 * np.pi) ** 12) * det)
    except:
        print("Matrix is not positive, semi-definite")
    if np.isnan(val):
        val = np.finfo(float).eps
    return val


"""initialize the emission, transition and initialisation matrices for HMM in chord recognition
PI - initialisation matrix, #A - transition matrix, #B - observation matrix"""


def initialize(chroma, templates, chords, nested_cof):

    """initialising PI with equal probabilities"""
    num_chords = len(chords)
    PI = np.ones(num_chords) / num_chords

    """initialising A based on nested circle of fifths"""
    eps = 0.01
    A = np.empty((num_chords, num_chords))
    for chord in chords:
        ind = nested_cof.index(chord)
        t = ind
        for i in range(num_chords):
            if t >= num_chords:
                t = t % num_chords
            A[ind][t] = (abs(num_chords // 2 - i) + eps) / (
                num_chords**2 + num_chords * eps
            )
            t += 1

    """initialising based on tonic triads - Mean matrix; Tonic with dominant - 0.8,
    tonic with mediant 0.6 and mediant-dominant 0.8, non-triad diagonal elements 
    with 0.2 - covariance matrix"""

    nFrames = np.shape(chroma)[1]
    B = np.zeros((num_chords, nFrames))
    meu_mat = np.zeros((num_chords, num_chords // 2))
    cov_mat = np.zeros((num_chords, num_chords // 2, num_chords // 2))
    meu_mat = np.array(templates)
    offset = 0

    for i in range(num_chords):
        if i == num_chords // 2:
            offset = 0
        tonic = offset
        if i < num_chords // 2:
            mediant = (tonic + 4) % (num_chords // 2)
        else:
            mediant = (tonic + 3) % (num_chords // 2)
        dominant = (tonic + 7) % (num_chords // 2)

        # weighted diagonal
        cov_mat[i, tonic, tonic] = 0.8
        cov_mat[i, mediant, mediant] = 0.6
        cov_mat[i, dominant, dominant] = 0.8

        # off-diagonal - matrix not positive semidefinite, hence determinant is negative
        # for n in [tonic,mediant,dominant]:
        #   for m in [tonic, mediant, dominant]:
        #       if (n is tonic and m is mediant) or (n is mediant and m is tonic):
        #           cov_mat[i,n,m] = 0.6
        #       else:
        #           cov_mat[i,n,m] = 0.8

        # filling non zero diagonals
        for j in range(num_chords // 2):
            if cov_mat[i, j, j] == 0:
                cov_mat[i, j, j] = 0.2
        offset += 1

    """observation matrix B is a multivariate Gaussian calculated from mean vector and 
    covariance matrix"""

    for m in range(nFrames):
        for n in range(num_chords):
            B[n, m] = multivariate_gaussian(
                chroma[:, m], meu_mat[n, :], cov_mat[n, :, :]
            )

    return (PI, A, B)


"""Viterbi algorithm to find Path with highest probability - dynamic programming"""


def viterbi(PI, A, B):
    (nrow, ncol) = np.shape(B)
    path = np.zeros((nrow, ncol))
    states = np.zeros((nrow, ncol))
    path[:, 0] = PI * B[:, 0]

    for i in range(1, ncol):
        for j in range(nrow):
            s = [(path[k, i - 1] * A[k, j] * B[j, i], k) for k in range(nrow)]
            (prob, state) = max(s)
            path[j, i] = prob
            states[j, i - 1] = state

    return (path, states)
